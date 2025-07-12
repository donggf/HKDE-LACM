import os
import random
import numpy as np
import optuna
from collections import defaultdict
import time
import tqdm
import joblib
import pandas as pd
from functools import reduce
from optuna.exceptions import TrialPruned
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from optuna import Study
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from optuna.distributions import BaseDistribution
from sklearn.feature_selection import VarianceThreshold , SelectKBest, f_classif,chi2
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.trial import TrialState
from optuna.trial import Trial
from optuna.samplers import BaseSampler
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

#Obtain the sample names and their corresponding labels.
def get_samples_and_labels(directory_name):
    directory = directory_name + "/train"
    # Get all the file names in the specified path.
    all_files_and_folders = os.listdir(directory)
    file_names = []
    for item in all_files_and_folders:
        if os.path.isfile(os.path.join(directory, item)):
            file_names.append(item)

    label_list = []
    filename_list = []
    for filename in file_names:
        filename = filename.split('.')[0]
        if filename == "" or filename == "kmer":
            continue
        if filename[0] == 'A':
            label_list.append("1")
            filename_list.append(filename)
        elif filename[0] == 'B':
            label_list.append("0")
            filename_list.append(filename)
        else:
            break
    print("filename_list:",filename_list,len(filename_list))
    print("label_list:",label_list,len(label_list))
    return filename_list,label_list

#Calculate the k-mers for each sample and aggregate them into a matrix.
def get_kmers(k,directory_name,target_name):
    directory = directory_name + "/train"
    # Calculate the k-mers
    mkdir_command = f"mkdir data/sample_data/train/kmer/{k}mer"
    os.system(mkdir_command)
    print(str(k) +"mer Directory created successfully.")
    print("Calculating k-mers:")
    for name in tqdm.tqdm(filename_list):
        file_name = name
        print("file_name:",file_name)
        #Calculate k-mers using Jellyfish.
        count_command = f"jellyfish count -m {k} -o {directory}/kmer/{k}mer/{file_name}.jf -c 3 -s 1G -t 16 {directory}/{file_name}.fa"
        os.system(count_command)
        time.sleep(4)

        # Export the results using `jellyfish dump` as a TSV file.
        dump_command = f"jellyfish dump -c -t {directory}/kmer/{k}mer/{file_name}.jf > {directory}/kmer/{k}mer/{file_name}.tsv"
        os.system(dump_command)
        time.sleep(2)

    all_reads = reduce(lambda x, y: [t + j for t in x for j in y], [['A', 'C', 'G', 'T']] * k)
    # Initialize an empty DataFrame to store the integrated results.
    merged_df = pd.DataFrame({
        'reads': all_reads,
        'frequency': [0] * len(all_reads)
    })
    i = 0
    folder_path = directory + '/kmer/' + str(k) + 'mer'
    #aggregate them into a matrix.
    for filename in tqdm.tqdm(filename_list):
        file_name = filename + ".tsv"
        filepath = os.path.join(folder_path, file_name)
        if os.path.isfile(filepath):
            sample = pd.read_csv(filepath, sep='\t', header=None,
                                 names=["reads", "frequency"])
            merged_df = pd.merge(merged_df, sample, on='reads', how='outer', suffixes=('_old','_new'))
            merged_df.rename(columns={'frequency_old': 'frequency','frequency_new': filename}, inplace=True)
            i = i + 1
    merged_df.drop('frequency', axis=1, inplace=True)
    merged_df.to_csv(target_name, sep="\t", index=False)
    print("The k-mers feature matrix has been saved at:",target_name)

#Read the embedded sequence.
def get_embedding(X_embedding_path,X_embedding_incorrect_path):
    X_embedding = pd.read_csv(X_embedding_path,sep="\t")
    X_embedding_incorrect = pd.read_csv(X_embedding_incorrect_path,sep="\t")
    print("Shape of the forward embedding vector:",X_embedding.shape)
    print("Shape of the reverse embedding vector:",X_embedding_incorrect.shape)
    return X_embedding,X_embedding_incorrect


# Custom Transformer: K-mer Pre-Filter
class KmerSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=50, kmer_dim=256, all_feature_names=None):
        self.k = k
        self.kmer_dim = kmer_dim
        self.selector = None
        self.selected_kmer_names = None
        self.all_feature_names = all_feature_names

    def fit(self, X, y=None):
        kmer_X = X[:, :self.kmer_dim]
        self.selector = SelectKBest(chi2, k=self.k).fit(kmer_X, y)

        if self.all_feature_names is not None:
            self.selected_kmer_names = [
                self.all_feature_names[i]
                for i in self.selector.get_support(indices=True)
            ]
            # print("self.selected_kmer_names:",self.selected_kmer_names)
        return self

    def transform(self, X):
        kmer_X = X[:, :self.kmer_dim]
        emb_X = X[:, self.kmer_dim:]
        kmer_selected = self.selector.transform(kmer_X)
        return np.hstack((kmer_selected, emb_X))


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method='variance', param=0.05):
        self.method = method
        self.param = param
        self.selector = None
        self.fitted_ = False

    def fit(self, X, y=None):
        if self.method == 'variance':
            self.selector = VarianceThreshold(threshold=self.param)
        elif self.method == 'kbest':
            self.selector = SelectKBest(score_func=f_classif, k=int(self.param))
        elif self.method == 'pca':
            self.selector = PCA(n_components=int(self.param))
        else:
            raise ValueError("Invalid feature selection method")

        self.selector.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("FeatureSelector is not fitted yet.")
        return self.selector.transform(X)

    def get_support(self, indices=False):
        if not self.fitted_:
            raise RuntimeError("FeatureSelector is not fitted yet.")
        if hasattr(self.selector, 'get_support'):
            return self.selector.get_support(indices=indices)
        else:
            raise AttributeError("Underlying selector does not support get_support().")


class FailureAwareDESampler(BaseSampler):
    def __init__(self, base_sampler, failure_vectors, all_param_names, radius=0.25):
        self.base_sampler = base_sampler
        self.failure_vectors = failure_vectors
        self.all_param_names = all_param_names
        self.radius = radius

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

    def sample_relative(self, study, trial, search_space):
        max_attempts = 50
        for _ in range(max_attempts):
            params = self.base_sampler.sample_relative(study, trial, search_space)
            x = np.array([params.get(k, 0.0) for k in self.all_param_names])
            if not self._is_in_failure_zone(x):
                return params
        print("[Warning] Unable to avoid the failed area, use the last sample")
        return params

    def infer_relative_search_space(self, study, trial):
        return self.base_sampler.infer_relative_search_space(study, trial)

    def _is_in_failure_zone(self, x):
        for fv in self.failure_vectors:
            if np.linalg.norm(x - fv) < self.radius:
                return True
        return False



class FailureRegionHandler:
    def __init__(self, main_param_names):
        self.main_param_names = main_param_names
        self._injected_failure_set = set()

    def _params_to_array(self, params: dict, all_param_names: list) -> np.ndarray:
        return np.array([
            float(params[k]) if (k in params and params[k] is not None) else 0.0
            for k in all_param_names
        ], dtype=np.float32)

    def _detect_failure_trials_grouped(self,trials, r=0.25, max_trials=8, min_improvement=1e-3, print_top_k=5):
        all_param_names = sorted({k for t in trials for k in t.params})

        trial_data = []
        for t in trials:
            p = t.params
            # Primary control key: Model type + Feature selector
            model_type = p.get('model_type_code')
            feature_selection = p.get('feature_selection_code')

            # Numeric vector recording the trial (excluding model_type and feature_selection)
            vec = [p.get(k, 0) for k in all_param_names if k not in {'model_type_code', 'feature_selection_code'}]
            trial_data.append({
                'trial': t,
                'vector': vec,
                'value': t.value,
                'group_key': (model_type, feature_selection)
            })

        # group processing
        groups = defaultdict(list)
        for d in trial_data:
            groups[d['group_key']].append(d)

        failure_trials = []

        for group_key, group_trials in groups.items():
            if len(group_trials) < max_trials:
                continue

            X = np.array([g['vector'] for g in group_trials])
            y = [g['value'] for g in group_trials]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            for i in range(len(X_scaled)):
                neighbors = [j for j in range(len(X_scaled))
                             if np.linalg.norm(X_scaled[i] - X_scaled[j]) <= r]
                if len(neighbors) >= max_trials:
                    local_values = [y[j] for j in neighbors]
                    if max(local_values) - min(local_values) < min_improvement:
                        failure_trials.append(group_trials[i]['trial'])

        print(f"\n[FailureRegionDetector] detected {len(failure_trials)} 个 failure trial")
        for i, t in enumerate(failure_trials[:print_top_k]):
            print(f"Failure Trial {i + 1}: {t.params}")
        return failure_trials

    def _inject_failure_region_trials(
            self,
            study,
            failure_trials,
            main_keys=('model_type_code', 'feature_selection_code'),
            param_distributions=None,
            n_perturb=5,
            n_random=5,
            perturb_std=0.05,
            set_pruned=True
    ):

        # ---- 1. Aggregation failed trial by main_keys ----
        def get_key(trial):
            return tuple(trial.params.get(k) for k in main_keys)

        grouped_trials = defaultdict(list)
        for t in failure_trials:
            grouped_trials[get_key(t)].append(t)

        injected = []

        for failure_key, trials in grouped_trials.items():

            # ---- 2. Select the central trial (with the smallest average distance to other trials)----
            def trial_to_vec(t):
                return np.array([
                    t.params.get(k, 0)
                    for k in param_distributions
                    if k not in main_keys
                ])

            vecs = np.array([trial_to_vec(t) for t in trials])
            if len(vecs) == 1:
                center_trial = trials[0]
            else:
                dists = np.sum([np.linalg.norm(vec - vecs, axis=1) for vec in vecs], axis=1)
                center_trial = trials[np.argmin(dists)]

            center_params = center_trial.params

            # ---- 3.Generate perturbation samples ----
            perturb_samples = []
            for _ in range(n_perturb):
                p = center_params.copy()
                for param, value in p.items():
                    if param in main_keys or param not in param_distributions:
                        continue
                    dist = param_distributions[param]

                    # Determine whether it is a perturbed interval parameter
                    if isinstance(dist, tuple) and len(dist) == 2 and all(isinstance(x, (int, float)) for x in dist):
                        low, high = dist
                        noise = np.random.normal(0, perturb_std * (high - low))
                        new_val = np.clip(value + noise, low, high)
                        if isinstance(value, int):
                            new_val = int(round(new_val))
                        p[param] = new_val

                    # If it is a discrete list (categorical), consider not perturbing or randomly sampling a new value
                    elif isinstance(dist, list):
                        # 1. Do not disturb, keep the original value
                        p[param] = value
                    # Other types are not processed
                    else:
                        p[param] = value
                perturb_samples.append(p)

            # ---- 4. Randomly generate samples matching key ----
            random_samples = []
            for _ in range(n_random):
                p = {k: v for k, v in zip(main_keys, failure_key)}
                for param, dist in param_distributions.items():
                    if param in main_keys:
                        continue

                    if isinstance(dist, tuple) and len(dist) == 2:
                        low, high = dist
                        if isinstance(low, int) and isinstance(high, int):
                            p[param] = random.randint(low, high)
                        else:
                            p[param] = random.uniform(low, high)
                    elif isinstance(dist, list):
                        p[param] = random.choice(dist)
                    else:
                        raise ValueError(f"Unknown distribution format for param: {param}")
                random_samples.append(p)

            # ---- 5. Merge and deduplicate injection ----
            all_samples = perturb_samples + random_samples

            def dedup_params(sample_list):
                seen = set()
                deduped = []
                for p in sample_list:
                    key = tuple(sorted(p.items()))
                    if key not in seen:
                        seen.add(key)
                        deduped.append(p)
                return deduped

            combined = dedup_params(all_samples)

            for p in combined:
                # Mark the field for identification processing in objective()
                p["force_fail"] = True
                study.enqueue_trial(p)
                injected.append(p)

        print(f"[Injection mechanism] Samples of injection failure areas: {len(injected)}")
        return len(injected)


class CyclicHybridSampler(BaseSampler):
    def __init__(self, warmup_trials=30, bo_patience=200, switch_threshold=0.001, seed=42):
        self.warmup_trials = warmup_trials
        self.bo_patience = bo_patience
        self.switch_threshold = switch_threshold
        self.seed = seed

        self.de_sampler = NSGAIISampler(seed=seed)
        self.bo_sampler = TPESampler(seed=seed,n_ei_candidates=100)
        self.current_sampler = self.de_sampler
        self.param_visit_counter = defaultdict(int)

        self._use_bo = False
        self._no_improve_counter = 0
        self._best_value = None
        self._last_de_round = 0

        self._bo_best_trial = None
        self._de_top_trials = []
        self._injected_failure_set = set()
        self.handler = FailureRegionHandler(main_param_names=["model_type_code", "feature_selection_code"])

    def infer_relative_search_space(self, study: Study, trial: Trial) -> Dict[str, BaseDistribution]:
        return self.current_sampler.infer_relative_search_space(study, trial)

    def sample_relative(self, study: Study, trial: Trial, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
        return self.current_sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study: Study, trial: Trial, param_name: str, param_distribution: BaseDistribution) -> Any:
        return self.current_sampler.sample_independent(study, trial, param_name, param_distribution)

    def _params_to_array(self, params, all_param_names):
        return np.array([float(params.get(name, 0.0) or 0.0) for name in all_param_names])


    def after_trial(self, study, trial, state, values):

        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if not hasattr(self, 'sampler_switch_log'):
            self.sampler_switch_log = []
        if not hasattr(self, 'sampler_trial_log'):
            self.sampler_trial_log = []

        self.sampler_trial_log.append({
            'trial': trial.number,
            'sampler': 'BO' if self._use_bo else 'DE'
        })

        if not self._use_bo and (trial.number - self._last_de_round) >= self.warmup_trials:
            print(f"\n[Switch] DE completed → Start BO (TPE)")

            failure_trials = self.handler._detect_failure_trials_grouped(
                study.trials,
                r=0.25,
                max_trials=6,
                min_improvement=1e-2
            )

            param_distributions = {
                "model_type_code": [0, 1, 2],  # rf, xgb, svm
                "feature_selection_code": [0, 1, 2],  # variance, kbest, pca
                "fs_param": [0.01, 0.05, 0.1, 0.2],

                # "fs_param_int": (50, 200),
                "fs_param_int": (10, 100),

                # RandomForest
                "n_estimators": (50, 200),
                "max_depth": [None, 10, 20],
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 4),

                # SVM
                "kernel_code": [0, 1, 2, 3],  # linear, rbf, sigmoid, poly
                "C": [0.1, 1, 10, 100],
                "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10],

                # XGBoost
                "xgb_n_estimators": (100, 300),
                "learning_rate": (0.01, 0.2),
                "xgb_max_depth": (3, 7),
                "subsample": (0.8, 1.0),
                "colsample_bytree": (0.8, 1.0),
            }

            #  2. The sampling point of the injection failure area is pruned trial
            injected_count = self.handler._inject_failure_region_trials(
                study=study,
                failure_trials=failure_trials,
                main_keys=('model_type_code', 'feature_selection_code'),
                param_distributions=param_distributions,
                n_perturb=20,
                n_random=5,
                set_pruned=True
            )

            print(f"[Sampler] Injected {injected_count} failure areas as pruned trials")

            self._de_top_trials = sorted(
                [t for t in complete_trials if t.number >= self._last_de_round],
                key=lambda t: t.value,
                reverse=True
            )[:10]

            self._use_bo = True
            self._best_value = study.best_value
            self._no_improve_counter = 0

            self.current_sampler = self.bo_sampler

            self.sampler_switch_log.append({
                'trial': trial.number,
                'from': 'DE',
                'to': 'BO',
                'reason': 'warmup_complete',
                'top_DE_trials': [(t.number, t.value) for t in self._de_top_trials]
            })

            if self._de_top_trials:
                print(f"[Information injection] Inject the DE optimal solution into the BO initial point")
                bo_startup_points = [t.params for t in self._de_top_trials if t.params]
                print(f"[Injection details] A total of {len(bo_startup_points)} points were injected")
                for t in self._de_top_trials:
                    print(f"  Trial #{t.number} | Value = {t.value:.5f}")

                for p in bo_startup_points:
                    study.enqueue_trial(p)

        if self._use_bo:
            if self._best_value is None:
                self._best_value = study.best_value
            elif study.best_value > self._best_value + self.switch_threshold:
                self._best_value = study.best_value
                self._no_improve_counter = 0
            else:
                self._no_improve_counter += 1

            if self._no_improve_counter >= self.bo_patience:
                print(f"\n[Escape from local optimum] No improvement after BO {self.bo_patience} times → Switch back to DE and explore again")

                # 1. Detection failure area
                failure_trials = self.handler._detect_failure_trials_grouped(
                    complete_trials,
                    r=0.25,
                    max_trials=10,
                    min_improvement=1e-3,
                    print_top_k=5
                )

                # 2. Vectorize and cache failed trials
                all_param_names = sorted({k for t in complete_trials for k in t.params})
                self._failure_vectors = np.array([
                    self.handler._params_to_array(t.params, all_param_names)
                    for t in failure_trials
                ])
                print(f"[Failure area injection] Inject {len(self._failure_vectors)} failure vectors in total")

                self.sampler_switch_log.append({
                    'trial': trial.number,
                    'from': 'BO',
                    'to': 'DE',
                    'reason': f'no_improvement_in_{self.bo_patience}_trials',
                    'bo_best_value': self._best_value,
                    'last_bo_trials': [(t.number, t.value) for t in complete_trials if t.number >= self._last_de_round]
                })

                self._use_bo = False
                self._no_improve_counter = 0
                self._best_value = study.best_value
                self._last_de_round = trial.number

                # self.de_sampler = NSGAIISampler(seed=self.seed)
                self.de_sampler = FailureAwareDESampler(
                    base_sampler=NSGAIISampler(seed=self.seed),
                    failure_vectors=self._failure_vectors,
                    all_param_names=all_param_names,
                    radius=0.25
                )
                self.current_sampler = self.de_sampler

        study.set_user_attr('sampler_trial_log', self.sampler_trial_log)
        study.set_user_attr('sampler_switch_log', self.sampler_switch_log)


# === Call optimization process ===
def run_cyclic_optimization(objective, n_trials=500,warmup_trials=20, bo_patience=15, switch_threshold=0.001):
    sampler = CyclicHybridSampler(warmup_trials=warmup_trials, bo_patience=bo_patience, switch_threshold=switch_threshold)
    early_stop_cb = GlobalEarlyStoppingCallback(no_improve_patience=70, switch_threshold=0.001)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[early_stop_cb], n_jobs=1)
    return study

# ========== Bayesian Optimization Objective Function ==========
def objective(trial):

    try:
        kmer_k = trial.suggest_int("kmer_topk",10,200)
        # ========== [Determine the injection failure area] ==========
        if trial.params.get("force_prune", False):
            print(f"[Skip] Trial {trial.number} belongs to the failed area, prune directly")
            raise TrialPruned()

        # ========== integer mapping ==========
        model_map = {0: "rf", 1: "xgb", 2: "svm"}
        fs_map = {0: "variance", 1: "kbest", 2: "pca"}

        model_type_code = trial.suggest_int("model_type_code", 0, 2)
        model_type = model_map[model_type_code]

        fs_code = trial.suggest_int("feature_selection_code", 0, 2)
        fs_method = fs_map[fs_code]

        # ========== Feature selection parameters ==========
        if fs_method == "variance":
            fs_param = trial.suggest_categorical("fs_param", [0.01, 0.05, 0.1, 0.2])
        elif fs_method == "kbest":
            fs_param = trial.suggest_int("fs_param_int", 5, min(X_train.shape[1], 200))
        elif fs_method == "pca":
            max_pca_components = min(int(X_train.shape[0] * 0.8), X_train.shape[1],200)
            # print("max_pca_components:",max_pca_components)
            fs_param = trial.suggest_int("fs_param_int", 5, max_pca_components)

        # ========== Model selection ==========
        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_categorical("max_depth", [None, 10, 20]),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                random_state=42
            )
        elif model_type == "svm":
            kernel_map = {0: "linear", 1: "rbf", 2: "sigmoid", 3: "poly"}
            kernel_code = trial.suggest_int("kernel_code", 0, 3)
            kernel = kernel_map[kernel_code]
            model = SVC(
                C=trial.suggest_categorical("C", [0.1, 1, 10, 100]),
                gamma=trial.suggest_categorical("gamma", [0.0001,0.001,0.01,0.1, 1, 10]),
                kernel=kernel,
                probability=True,
                random_state=42,
                max_iter=5000,  #
            )
        else:  # XGBoost
            model = XGBClassifier(
                n_estimators=trial.suggest_int("xgb_n_estimators", 100, 300),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
                max_depth=trial.suggest_int("xgb_max_depth", 3, 7),
                subsample=trial.suggest_float("subsample", 0.8, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.8, 1.0),
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                random_state=42
            )


        kmer_columns = X_columns[:kmer_dim]
        # print("kmer_columns:",kmer_columns)

        pipeline = Pipeline([
            ('kmer_filter', KmerSelector(k=kmer_k, kmer_dim=kmer_dim, all_feature_names=kmer_columns)),
            ('scaler', StandardScaler()),
            ('feature_selection', FeatureSelector(method=fs_method, param=fs_param)),
            ('classifier', model)
        ])

        score_list = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='accuracy')
        mean_score = score_list.mean()

        trial.report(mean_score, step=0)

        return mean_score

    except TrialPruned:
        raise



class GlobalEarlyStoppingCallback:
    def __init__(self, no_improve_patience=200, switch_threshold=0.001):
        self.patience = no_improve_patience
        self.threshold = switch_threshold
        self.best_value = None
        self.no_improve_counter = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # 初始化最优值
        if self.best_value is None:
            self.best_value = study.best_value
            self.no_improve_counter = 0
            return

        if study.best_value > self.best_value + self.threshold:
            self.best_value = study.best_value
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1

        # Stop condition is met
        if self.no_improve_counter >= self.patience:
            print(f"\n[Terminate optimization] No performance improvement for {self.patience} consecutive times, stop optimization early")
            study.stop()


if __name__ == "__main__":
    current_directory = os.getcwd()
    # print("Current directory:", current_directory)
    directory_name = current_directory + "/data/sample_data"
    # print("directory:",directory_name)
    filename_list, label_list = get_samples_and_labels(directory_name)
    k = int(input("Please enter the value of k："))
    target_name = directory_name + "/result/sample_data_" + str(k) + "mer.tsv"
    get_kmers(k,directory_name,target_name)

    X_NULL = pd.DataFrame({'name': filename_list})
    X = pd.read_csv(target_name, sep="\t", header=None)
    print("X:", X)
    y = pd.DataFrame(label_list)
    y = y.astype(int)
    print("y", y, y.shape)
    X_embedding, X_embedding_incorrect = get_embedding(
        "data/sample_data/train/result/merged_correct_sequence_embeddings_1024.tsv",
        "data/sample_data/train/result/merged_incorrect_sequence_embeddings_1024.tsv")

    # Basic processing of sequences.
    X.fillna(0, inplace=True)
    X = X.T
    X.columns = X.iloc[0, :]
    X = X.iloc[1:]
    X.columns.values[0] = 'name'

    X_kmer = pd.merge(X_NULL, X, on='name', how='left')
    X = X_kmer
    print("X:", X)

    X_name = X.iloc[:, 0]
    X = X.iloc[:, 1:].astype(float)
    # Convert the frequency count to relative frequency.
    mean_values = X.mean()
    sorted_mean_values = sorted(mean_values)
    index_mean_value = sorted_mean_values[int((len(sorted_mean_values) / 5) * 4)]
    # print("index_mean_value:",index_mean_value)
    X_1 = X
    X_1 = X_1.drop(columns=mean_values[mean_values < index_mean_value].index)
    X = X_1

    with open('data/sample_data/result/column_names.txt', 'w') as file:
        for column in X.columns:
            file.write(column + '\n')
    print("Column name file saved")

    print("After removing rare k-mers, the number of features is::", len(X.columns))

    # Convert frequency counts to frequencies.
    X_sum = X.sum(axis=1)  
    X_new = (X.T / X_sum).T
    X = X_new
    X.fillna(0, inplace=True)
    kmer_dim = X.shape[1]
    print("kmer_dim:", kmer_dim)

    X = pd.concat([X_name, X], axis=1)

    print("X:", X, type(X), X.shape)
    print("X_embedding", X_embedding)

    merged_data = pd.merge(X, X_embedding, on='name', how='left')
    merged_data_2 = pd.merge(merged_data, X_embedding_incorrect, on='name', how='left')

    X = merged_data_2
    X.iloc[:, -1024:] = -X.iloc[:, -1024:]
    X.index = X['name']
    X = X.iloc[:, 1:]
    X_columns = X.columns
    X.fillna(0, inplace=True)
    print("The final feature matrix:", X)

    y = y.values.ravel()  # Check the shape of y_train and reshape it to a one-dimensional array
    X_train = X
    X_train = X_train.values
    y_train = y

    # ========== Start Bayesian Optimization ==========
    start_time = time.time()
    study = optuna.create_study(direction="maximize")
    study = run_cyclic_optimization(objective, n_trials=2000,warmup_trials=20, bo_patience=30, switch_threshold=0.001)

    saved_configs = []


    def is_unique_config(new_params, saved_configs):
        for cfg in saved_configs:
            same_fs = new_params['feature_selection'] == cfg['feature_selection']
            same_model = new_params['model_type'] == cfg['model_type']
            if same_fs and same_model:
                return False
        return True


    saved_count = 0
    i = 0
    max_to_save = 5
    top_trials_sorted = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True
    )

    while saved_count < max_to_save and i < len(top_trials_sorted):
        trial = top_trials_sorted[i]
        params = trial.params


        model_map = {0: "rf", 1: "xgb", 2: "svm"}
        fs_map = {0: "variance", 1: "kbest", 2: "pca"}

        model_type_code = trial.suggest_int("model_type_code", 0, 2)
        model_type = model_map[model_type_code]

        fs_code = trial.suggest_int("feature_selection_code", 0, 2)
        fs_method = fs_map[fs_code]

        if fs_method == "variance":
            fs_param = params["fs_param"]
        else:
            fs_param = params["fs_param_int"]

        params = params.copy()
        params["feature_selection"] = fs_method
        params["model_type"] = model_type

        if not is_unique_config(params, saved_configs):
            i += 1
            continue  # Skip repeated combinations

        # Save configuration
        saved_configs.append({
            'feature_selection': params['feature_selection'],
            'model_type': params['model_type']
        })

        # Building a feature selector
        fs_selector = FeatureSelector(method=params["feature_selection"], param=fs_param)
        kmer_k_selector = KmerSelector(k=params["kmer_topk"], kmer_dim=kmer_dim)
        # Build a classifier
        if params["model_type"] == "rf":
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42
            )
        elif params["model_type"] == "svm":
            kernel_map = {0: "linear", 1: "rbf", 2: "sigmoid", 3: "poly"}
            kernel_code = trial.suggest_int("kernel_code", 0, 3)
            kernel = kernel_map[kernel_code]
            model = SVC(
                C=params["C"],
                gamma=params["gamma"],
                kernel=kernel,
                probability=True,
                random_state=42,
                max_iter=5000,
            )
        else:
            model = XGBClassifier(
                n_estimators=params["xgb_n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["xgb_max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )

        kmer_columns = X_columns[:kmer_dim]
        pipeline = Pipeline([
            ('kmer_filter', KmerSelector(k=params["kmer_topk"], kmer_dim=kmer_dim,all_feature_names=kmer_columns)),
            ('scaler', StandardScaler()),
            ('feature_selection', fs_selector),
            ('classifier', model)
        ])


        # Fitting the entire training set
        print(pipeline)
        pipeline.fit(X_train, y_train)

        # Evaluation model
        accuracy_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='accuracy')
        mean_accuracy = accuracy_scores.mean()

        # Export and save the model
        print(f"\n### Model {saved_count + 1} (accuracy {mean_accuracy:.4f})###")
        print("parameter:", params)
        print(f"feature selection: {params['feature_selection']} | model type: {params['model_type']}")
        
        dir_path = Path(f"model_{k}")
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"model_{k}/model_top_{saved_count + 1}.pkl"
        joblib.dump(pipeline, filename)
        print(f"Model saved as {filename}")

        saved_count += 1
        i += 1

        if i >= len(top_trials_sorted):
            print("All trials have been traversed and no sufficient models were found.")
            break

        end_time = time.time()
        training_time = end_time - start_time
        print("training time：",training_time)

