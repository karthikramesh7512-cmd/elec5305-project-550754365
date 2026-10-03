"""Feature unit tests and independent saved-SVM provenance checks."""
import unittest
import json
import inspect
from unittest.mock import patch
import numpy as np
import pandas as pd
import svm_baseline as svm
from extract_features import features_from_image, FEATURES


class Tests(unittest.TestCase):
    def test_fixed_roi_dc_and_energy(self):
        image=np.zeros((480,7168))
        image[:,1100:3100]=5*np.sin(2*np.pi*np.arange(2000)/100)+np.arange(480)[:,None]
        a=features_from_image(image)
        self.assertAlmostEqual(a['max_envelope'],5,places=10)
        self.assertAlmostEqual(a['rf_rms'],5/np.sqrt(2),places=10)
        self.assertAlmostEqual(a['rf_energy'],960000*a['rf_rms']**2,places=6)
        image[:,:1100]=1e8;image[:,3100:]=-1e8;image[:,1100:3100]+=np.arange(480)[:,None]*2
        b=features_from_image(image)
        np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=1e-10,atol=1e-8)
        self.assertEqual(list(inspect.signature(features_from_image).parameters),['image'])
        self.assertTrue(np.isfinite(list(features_from_image(np.zeros((480,7168))).values())).all())

    def test_reject_test_rows_at_fit_or_selection(self):
        frame=pd.DataFrame({'sample_id':['a','b'],'has_flaw':[0,1],'partition':['test','test'],
                            **{name:[0.,1.] for name in FEATURES}})
        with self.assertRaises(ValueError):svm.fit_select(frame,frame)
        train=frame.copy();train['partition']='train'
        with self.assertRaises(ValueError):svm.fit_select(train,frame)

    def test_fit_calls_receive_training_only(self):
        # Synthetic test only; the seven saved experiment models are not refitted.
        train=pd.DataFrame({'sample_id':[f't{i}' for i in range(8)],
            'has_flaw':[0]*4+[1]*4,'partition':'train',
            **{name:np.arange(8,dtype=float)+(k*.1) for k,name in enumerate(FEATURES)}})
        val=pd.DataFrame({'sample_id':['v0','v1'],'has_flaw':[0,1],'partition':'validation',
            **{name:[100.+k,-100.-k] for k,name in enumerate(FEATURES)}})
        pipeline_calls=[];svc_calls=[]
        original_pipeline_fit=svm.Pipeline.fit
        original_svc_fit=svm.SVC.fit
        def fit_pipeline(instance,x,y,*args,**kwargs):
            pipeline_calls.append((x.copy(),y.copy()))
            return original_pipeline_fit(instance,x,y,*args,**kwargs)
        def fit_svc(instance,x,y,*args,**kwargs):
            svc_calls.append((np.asarray(x).copy(),np.asarray(y).copy()))
            return original_svc_fit(instance,x,y,*args,**kwargs)
        with patch.object(svm.Pipeline,'fit',fit_pipeline),patch.object(svm.SVC,'fit',fit_svc):
            svm.fit_select(train,val,candidates=[dict(kernel='linear',C=.01,gamma='scale'),
                                                 dict(kernel='rbf',C=.1,gamma=1)])
        self.assertEqual(len(pipeline_calls),2);self.assertEqual(len(svc_calls),2)
        expected=(train[FEATURES]-train[FEATURES].mean())/train[FEATURES].std(ddof=0)
        for (x,y),(scaled,svc_y) in zip(pipeline_calls,svc_calls):
            pd.testing.assert_frame_equal(x,train[FEATURES])
            np.testing.assert_array_equal(y,train.has_flaw)
            np.testing.assert_allclose(scaled,expected)
            np.testing.assert_array_equal(svc_y,train.has_flaw)

    def test_saved_models_and_artifacts(self):
        ordinary=pd.read_csv(svm.ROOT/'data/splits/ordinary_split.csv')
        holdout=pd.read_csv(svm.ROOT/'data/splits/source_holdout_folds.csv')
        features=pd.read_csv(svm.ROOT/'data/features_experimental.csv')
        provenance=pd.read_csv(svm.OUT/'fitting_and_selection_ids.csv')
        predictions=pd.read_csv(svm.OUT/'per_sample_results.csv')
        summary=pd.read_csv(svm.OUT/'svm_summary.csv')
        grid=pd.read_csv(svm.OUT/'validation_grid.csv')
        self.assertEqual(len(predictions),49000)
        self.assertEqual(len(grid),175)
        for scheme,fold,manifest in [('ordinary',0,ordinary)]+[('source_holdout',int(f),m) for f,m in holdout.groupby('fold')]:
            frame=manifest.merge(features.drop(columns='batch'),on='sample_id',validate='one_to_one')
            train=frame[frame.partition.eq('train')];val=frame[frame.partition.eq('validation')]
            test=frame[frame.partition.eq('test')]
            roles=provenance[provenance.scheme.eq(scheme)&provenance.fold.eq(fold)]
            self.assertEqual(set(roles[roles.role.eq('scaler_and_svc_fit')].sample_id),set(train.sample_id))
            self.assertEqual(set(roles[roles.role.eq('model_selection')].sample_id),set(val.sample_id))
            self.assertFalse(set(roles.sample_id)&set(test.sample_id))
            model=svm.joblib.load(svm.OUT/f'{scheme}_{fold}_pipeline.joblib')
            scaler=model.named_steps['scaler']
            self.assertEqual(int(scaler.n_samples_seen_),len(train))
            np.testing.assert_allclose(scaler.mean_,train[FEATURES].mean(),rtol=1e-12,atol=1e-9)
            np.testing.assert_allclose(scaler.var_,train[FEATURES].var(ddof=0),rtol=1e-10,atol=1e-9)
            self.assertEqual(list(scaler.feature_names_in_),FEATURES)
            svc=model.named_steps['svc']
            self.assertEqual(svc.shape_fit_,(len(train),len(FEATURES)))
            np.testing.assert_allclose(svc.support_vectors_,scaler.transform(train[FEATURES])[svc.support_],
                                       rtol=1e-12,atol=1e-12)
            if scheme=='source_holdout':self.assertFalse(train.source_label.eq(test.test_source.iloc[0]).any())
            result=predictions[predictions.scheme.eq(scheme)&predictions.fold.eq(fold)]
            self.assertEqual(len(result),7000);self.assertTrue(result.sample_id.is_unique)
            result=result[result.partition.eq('test')].set_index('sample_id').loc[test.sample_id]
            self.assertEqual(set(result.index),set(test.sample_id))
            np.testing.assert_array_equal(result.predicted_label,model.predict(test[FEATURES]))
            np.testing.assert_allclose(result.decision_score,model.decision_function(test[FEATURES]),rtol=1e-12,atol=1e-12)
            met=svm.metrics(result.has_flaw,result.decision_score,result.predicted_label)
            saved=summary[summary.scheme.eq(scheme)&summary.fold.eq(fold)].iloc[0]
            for key,value in met.items():self.assertAlmostEqual(saved[key],value,places=12)
            precision,recall,_=svm.precision_recall_curve(result.has_flaw,result.decision_score)
            self.assertAlmostEqual(saved.PR_AUC,svm.auc(recall,precision),places=12)
            self.assertAlmostEqual(saved.average_precision,
                svm.average_precision_score(result.has_flaw,result.decision_score),places=12)
            candidates=grid[grid.scheme.eq(scheme)&grid.fold.eq(fold)]
            winner=candidates.sort_values(['validation_balanced_accuracy','candidate'],ascending=[False,True]).iloc[0]
            self.assertEqual(saved.candidate,winner.candidate)
        checks=json.loads((svm.OUT/'checks.json').read_text())
        for path,value in checks['protected_sha256'].items():
            if path == 'dsp_baseline.py':
                # Verify the original source hash after undoing only relocation.
                import hashlib
                original = (svm.ROOT/'src/dsp_baseline.py').read_text().replace(
                    'Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')
                self.assertEqual(hashlib.sha256(original.encode('utf-8')).hexdigest(),value)
            else:
                self.assertEqual(svm.sha(svm.ROOT/path.replace(chr(92), "/")),value)


if __name__=='__main__':unittest.main()
