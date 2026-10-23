"""Separate historical signal experiments. Never reads/writes forward decisions."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from accounting import raw_samples,simulate
from market_data import read_snapshot
from research import FEATURES,backtest,ridge_predict
from daily import recent_config


def sequences(context,rows,lookback=20):
    positions=context.index.get_indexer(rows.index)
    if (positions<lookback-1).any():
        raise ValueError('Insufficient strictly historical feature context')
    values=context[FEATURES].to_numpy(dtype=np.float32)
    return np.stack([values[i-lookback+1:i+1] for i in positions])


def fit_lstm(context,train,predict_rows,seed,epochs=20,validation=None):
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        pass  # A preceding independent fit already initialized the runtime.
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    x=sequences(context,train)
    mean,scale=x.mean(axis=(0,1)),x.std(axis=(0,1));scale=np.where(scale>0,scale,1.)
    target_mean=float(train.target.mean());target_scale=max(float(train.target.std()),1e-8)
    options=tf.data.Options();options.threading.private_threadpool_size=2
    def dataset(rows):
        x=(sequences(context,rows)-mean)/scale
        y=((rows.target.to_numpy()-target_mean)/target_scale).astype(np.float32)
        return tf.data.Dataset.from_tensor_slices((x,y)).batch(32).with_options(options)
    model=tf.keras.Sequential([tf.keras.layers.Input((20,len(FEATURES))),
        tf.keras.layers.LSTM(16,kernel_regularizer=tf.keras.regularizers.l2(.001)),tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
    callbacks=[] if validation is None else [tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True)]
    history=model.fit(dataset(train),validation_data=dataset(validation) if validation is not None else None,
                      epochs=epochs,callbacks=callbacks,verbose=0,shuffle=False)
    prediction=model.predict(dataset(predict_rows),verbose=0).ravel()*target_scale+target_mean
    best=int(np.argmin(history.history['val_loss']))+1 if validation is not None else epochs
    return prediction,best


def choose_threshold(validation,prediction,config):
    candidates=[]
    for threshold in (0.,.001,.003,float('inf')):
        ledger=backtest(validation,(prediction>threshold)*config.max_exposure,config)
        candidates.append((float(ledger.equity.iloc[-1]),threshold))
    return max(candidates,key=lambda x:(x[0],x[1]))


def fit_tree(train,rows,leaves,regularization):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaves,min_samples_leaf=50,
        l2_regularization=regularization,early_stopping=False,random_state=42)
    model.fit(train[FEATURES],train.target)
    return model.predict(rows[FEATURES])


def run(snapshot,out,include_lstm=False,seeds=(7,42,123)):
    config=recent_config()
    bars,actions,manifest=read_snapshot(snapshot,pd.Timestamp.now(tz='UTC'),allow_stale=True)
    context=raw_samples(bars,actions,include_unlabeled=True)
    frame=context.iloc[19:].dropna()
    cutoff=pd.Timestamp(config.holdout_start)
    evaluation=frame[frame.exit_date<cutoff].iloc[config.min_train+config.validation+2:]
    if evaluation.empty:
        raise ValueError('Insufficient pre-holdout development history')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protocol={'schema_version':1,'snapshot_sha256':manifest['sha256'],'config':asdict(config),'features':FEATURES,
              'lstm':include_lstm,'lstm_seeds':list(seeds) if include_lstm else [],'lookback':20,
              'ridge_alphas':[1.,100.,10000.],'tree_settings':[[7,1.],[15,10.]],'lstm_max_epochs':20,
              'first_evaluation':str(evaluation.index[0].date()),'last_evaluation':str(evaluation.index[-1].date()),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'purpose':'Historical development only. No automatic promotion or changes to forward experiment.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    names=['ridge','gradient_boosting']+([f'lstm_seed_{s}' for s in seeds] if include_lstm else [])
    predictions={name:[] for name in names};choices=[]
    for start in range(0,len(evaluation),config.fold):
        test=evaluation.iloc[start:start+config.fold]
        history=frame[(frame.index<test.index[0]) & (frame.exit_date<=test.index[0])]
        validation=history.iloc[-config.validation:]
        train=history.iloc[:-config.validation]
        train=train[train.exit_date<=validation.index[0]]
        if len(train)<config.min_train:
            raise ValueError('Insufficient purged training history')
        for name in names:
            if name=='ridge':
                candidates=[]
                for alpha in (1.,100.,10000.):
                    wealth,threshold=choose_threshold(validation,ridge_predict(train,validation,alpha),config)
                    candidates.append((wealth,threshold,alpha))
                _,threshold,alpha=max(candidates,key=lambda x:(x[0],x[1],x[2]))
                prediction=ridge_predict(history,test,alpha);settings={'alpha':alpha}
            elif name=='gradient_boosting':
                candidates=[]
                for leaves,regularization in ((7,1.),(15,10.)):
                    wealth,threshold=choose_threshold(validation,fit_tree(train,validation,leaves,regularization),config)
                    candidates.append((wealth,threshold,leaves,regularization))
                _,threshold,leaves,regularization=max(candidates,key=lambda x:(x[0],x[1],-x[2]))
                prediction=fit_tree(history,test,leaves,regularization);settings={'leaves':leaves,'l2':regularization}
            else:
                seed=int(name.rsplit('_',1)[1])
                val_prediction,epochs=fit_lstm(context,train,validation,seed,validation=validation)
                _,threshold=choose_threshold(validation,val_prediction,config)
                prediction,_=fit_lstm(context,history,test,seed,epochs=epochs)
                settings={'seed':seed,'refit_epochs':epochs}
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite candidate prediction')
            result=test.copy();result['prediction']=prediction
            result['desired']=(prediction>threshold)*config.max_exposure
            predictions[name].append(result)
            choices.append({'model':name,'first_signal':str(test.index[0].date()),'last_signal':str(test.index[-1].date()),
                'last_training_label_exit':str(history.exit_date.max().date()),'threshold':None if np.isinf(threshold) else threshold,
                **settings})
            print(name,str(test.index[0].date()),'completed',flush=True)
    subset=bars[(bars.session>=evaluation.index[0]) & (bars.session<=evaluation.exit_date.iloc[-1])]
    events=actions[actions.effective_date.isin(subset.session)]
    strategies={name:pd.concat(parts) for name,parts in predictions.items()}
    summaries={}
    for name in names+['cash','buy_hold','momentum_20']:
        rows=strategies[name] if name in strategies else evaluation
        if name in strategies:
            desired=rows.desired
        elif name=='cash':desired=pd.Series(0.,index=rows.index)
        elif name=='buy_hold':desired=pd.Series(config.max_exposure,index=rows.index)
        else:desired=(rows.return_20>0)*config.max_exposure
        ledger,fills=simulate(subset,events,desired,config,manifest['observed_at'],stop=name in strategies)
        ledger.to_csv(out/f'{name}_ledger.csv',index=False);fills.to_csv(out/f'{name}_fills.csv',index=False)
        summaries[name]={'total_return':float(ledger.equity.iloc[-1]/10000-1),'max_drawdown':float(ledger.drawdown.min()),
            'cost_dollars':float(ledger.cost.sum()),'stop_triggered':bool(ledger.halted.any()),'fills':len(fills)}
        if name in strategies:
            rows.to_csv(out/f'{name}_predictions.csv')
            summaries[name]['forecast_mae']=float(np.abs(rows.prediction-rows.target).mean())
            summaries[name]['zero_forecast_mae']=float(np.abs(rows.target).mean())
    report={'protocol':protocol,'choices':choices,'results':summaries,'forward_experiment_changed':False,
            'limitations':['Historical development results are subject to model-selection and asset-selection bias.',
                          '2026 performance holdout excluded. No current/future predictions were changed or promoted.',
                          'Candidate evaluation uses common signal dates; differs from earlier reports by 20-row sequence warmup.',
                          'Model selection uses validation return/cost proxy; final report uses raw-share ledger with payment-date cash.',
                          'SIP action completeness still needs independent corroboration. Costs and fills are modeled.',
                          'Seed results must be inspected together, not cherry-picked. Test periods must not drive ongoing tuning.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--include-lstm',action='store_true')
    args=parser.parse_args();report=run(args.snapshot,args.out,args.include_lstm)
    print(pd.DataFrame(report['results']).T.to_string())


if __name__=='__main__':
    main()
