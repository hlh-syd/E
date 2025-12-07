
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os
import gc
import sys

# Add project root to path to import dataset.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from dataset import qc_snp, qc_by_missing, merge_data, impute_with_uncertainty
except ImportError:
    # Fallback if running from scripts/ subdir
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataset import qc_snp, qc_by_missing, merge_data, impute_with_uncertainty

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for scikit-survival
try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Warning: scikit-survival not installed. LASSO-Cox functionality will be limited.")

class TraditionalAnalysis:
    def __init__(self, data_dir="e:\\TERM\\data", output_dir="e:\\TERM\\results"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Initializing Traditional Analysis pipeline...")
        print(f"Data Directory: {data_dir}")
        print(f"Output Directory: {output_dir}")
        
        # 1. Load & Merge Data using dataset.py logic
        self.df = self._load_and_preprocess_data()
        
        # 2. Define Clinical vs Feature columns
        self.clinical_patterns = ['age', 'gender', 'stage', 'T', 'E', 'status', 'time', 'risk', 'cluster']
        self.clinical_cols = [c for c in self.df.columns if any(p in c.lower() for p in self.clinical_patterns)]
        
        # Ensure T and E are numeric and exist
        if 'T' not in self.df.columns or 'E' not in self.df.columns:
            # Try to map common names
            col_map = {}
            for c in self.df.columns:
                cl = c.lower()
                if 'time' in cl or 'duration' in cl: col_map[c] = 'T'
                if 'status' in cl or 'event' in cl or 'dead' in cl: col_map[c] = 'E'
            self.df.rename(columns=col_map, inplace=True)
            
        if 'T' not in self.df.columns or 'E' not in self.df.columns:
             raise ValueError("Could not find survival time (T) or event (E) columns after merging.")

        self.df['T'] = pd.to_numeric(self.df['T'], errors='coerce')
        self.df['E'] = pd.to_numeric(self.df['E'], errors='coerce')
        self.df = self.df.dropna(subset=['T', 'E'])
        
        self.snp_cols = [c for c in self.df.columns if c not in self.clinical_cols]
        print(f"Dataset Loaded: {self.df.shape[0]} samples, {len(self.snp_cols)} SNPs/Features.")

    def _load_and_preprocess_data(self):
        """
        Calls dataset.py functions to merge, qc and impute data.
        """
        # 1. Merge Data (Expression/SNP + Clinical + Survival)
        # Assuming dataset.py's merge_data handles the basic join
        print("Merging data from raw files...")
        merged_df = merge_data(self.data_dir, missing_threshold=0.2)
        
        # 2. QC
        print("Running QC (Missingness + SNP specific)...")
        # Identify feature columns for QC (exclude clinical-like)
        # This is tricky before we fully define them, but qc_by_missing handles it by ratio
        # Let's run generic missingness QC first
        df_qc = qc_by_missing(merged_df, sample_threshold=0.05, feature_threshold=0.05)
        
        # SNP QC (MAF, HWE)
        # qc_snp in dataset.py expects the full dataframe and detects features automatically
        # We need to ensure it treats our data as SNP data (0,1,2)
        print("Running HWE/MAF filtering...")
        df_qc = qc_snp(df_qc, maf_threshold=0.01, hwe_threshold=0.01)
        
        # 3. Imputation (Optional but recommended for traditional analysis too)
        # Using simple imputation or Soft-GAIN?
        # Traditional analysis often uses simple imputation or complete case.
        # But user requested "dataset.py process".
        # Let's check if we have missing values left
        if df_qc.isna().sum().sum() > 0:
            print("Imputing missing values (using Soft-GAIN from dataset.py)...")
            # impute_with_uncertainty returns (df_imputed, df_std)
            df_imp, _ = impute_with_uncertainty(df_qc, n_runs=1) # 1 run for speed in traditional pipeline
            return df_imp
        else:
            return df_qc

    def plot_km(self, variable, output_path=None, discrete=True):
        """
        Plots KM curve. 
        If discrete=True, plots by unique values (0,1,2).
        If discrete=False, splits by median.
        """
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(8, 6))
        
        data = self.df[[variable, 'T', 'E']].dropna()
        
        if discrete:
            # Check if truly discrete (few unique values)
            # If float but few values, round them
            if data[variable].nunique() <= 5:
                data['Group'] = data[variable].round().astype(int)
                groups = sorted(data['Group'].unique())
                for g in groups:
                    mask = data['Group'] == g
                    if mask.sum() > 0:
                        kmf.fit(data.loc[mask, 'T'], event_observed=data.loc[mask, 'E'], label=f"{variable}={g} (n={mask.sum()})")
                        kmf.plot_survival_function(ci_show=False, linewidth=2)
                
                # Multivariate logrank
                if len(groups) > 1:
                    res = multivariate_logrank_test(data['T'], data['Group'], data['E'])
                    p_val = res.p_value
                else:
                    p_val = 1.0
            else:
                # Fallback to median split if too many values
                discrete = False
        
        if not discrete:
            median_val = data[variable].median()
            mask_high = data[variable] > median_val
            mask_low = data[variable] <= median_val
            
            kmf.fit(data.loc[mask_high, 'T'], event_observed=data.loc[mask_high, 'E'], label=f"High (> {median_val:.2f})")
            kmf.plot_survival_function(ci_show=False, linewidth=2)
            kmf.fit(data.loc[mask_low, 'T'], event_observed=data.loc[mask_low, 'E'], label=f"Low (<= {median_val:.2f})")
            kmf.plot_survival_function(ci_show=False, linewidth=2)
            
            res = logrank_test(data.loc[mask_high, 'T'], data.loc[mask_low, 'T'], 
                               data.loc[mask_high, 'E'], data.loc[mask_low, 'E'])
            p_val = res.p_value

        plt.title(f"KM Curve for {variable}\np={p_val:.4e}")
        plt.xlabel("Time (Days)")
        plt.ylabel("Survival Probability")
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Saved KM plot to {output_path}")
        else:
            plt.show()

    def univariate_cox(self):
        """
        Runs univariate Cox for SNPs assuming Additive Model (0, 1, 2 as continuous).
        """
        print("\n=== Step 1: Univariate Cox Screening ===")
        results = []
        
        total_snps = len(self.snp_cols)
        print(f"Screening {total_snps} features...")
        
        for i, snp in enumerate(self.snp_cols):
            if i % 500 == 0:
                print(f"  Processed {i}/{total_snps}...", flush=True)
                gc.collect()
                
            try:
                subset = self.df[[snp, 'T', 'E']].dropna()
                if subset[snp].std() == 0: continue
                
                cph = CoxPHFitter()
                cph.fit(subset, duration_col='T', event_col='E')
                
                summ = cph.summary.loc[snp]
                results.append({
                    'snp': snp,
                    'coef': summ['coef'],
                    'HR': np.exp(summ['coef']),
                    'p_value': summ['p'],
                    'lower_95': np.exp(summ['coef lower 95%']),
                    'upper_95': np.exp(summ['coef upper 95%'])
                })
            except:
                continue
                
        res_df = pd.DataFrame(results)
        
        if not res_df.empty:
            # FDR Correction
            reject, pvals_adj, _, _ = multipletests(res_df['p_value'], alpha=0.05, method='fdr_bh')
            res_df['p_adj'] = pvals_adj
            res_df = res_df.sort_values('p_value')
            
            out_file = os.path.join(self.output_dir, "snp_univariate_results.csv")
            res_df.to_csv(out_file, sep='\t', index=False)
            print(f"Saved univariate results to {out_file}")
            
            # Select candidates (P < 0.05)
            candidates = res_df[res_df['p_value'] < 0.05]['snp'].tolist()
            print(f"Found {len(candidates)} features with p < 0.05.")
            return candidates, res_df
        else:
            print("No features converged in univariate Cox.")
            return [], res_df

    def lasso_cox(self, candidate_snps):
        """
        Runs LASSO-Cox to select final markers.
        """
        print("\n=== Step 2: LASSO-Cox Feature Selection ===")
        if not SKSURV_AVAILABLE or len(candidate_snps) < 2:
            print("Skipping LASSO (Not enough candidates or sksurv missing).")
            return candidate_snps[:10] # Return top 10 as fallback
            
        X = self.df[candidate_snps]
        y = np.array([(bool(e), t) for e, t in zip(self.df['E'], self.df['T'])], 
                     dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        
        print("Estimating optimal alpha via CV...")
        coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, fit_baseline_model=True)
        coxnet.fit(X, y)
        
        # Auto-select alpha that gives reasonable sparsity (5-30 features)
        alphas = coxnet.alphas_
        best_alpha = alphas[-1] # Default to simplest
        
        for alpha in alphas:
            model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha], fit_baseline_model=True)
            model.fit(X, y)
            n_features = np.sum(model.coef_ != 0)
            if 5 <= n_features <= 30:
                best_alpha = alpha
                break 
        
        print(f"Selected alpha: {best_alpha}")
        
        final_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best_alpha], fit_baseline_model=True)
        final_model.fit(X, y)
        
        coefs = pd.Series(final_model.coef_.ravel(), index=X.columns)
        selected = coefs[coefs != 0]
        
        print(f"LASSO selected {len(selected)} features.")
        selected.to_csv(os.path.join(self.output_dir, "lasso_selected_features.csv"), sep='\t')
        return selected

    def build_final_model(self, selected_features_series):
        """
        Calculates Risk Score, plots KM/ROC, and Correlation plots.
        """
        print("\n=== Step 3: Final Model & Evaluation ===")
        
        # 1. Calculate Risk Score
        risk_score = np.zeros(len(self.df))
        for feat, coef in selected_features_series.items():
            risk_score += self.df[feat] * coef
            
        self.df['RiskScore'] = risk_score
        
        # 2. Plot Risk Score KM
        self.plot_km('RiskScore', output_path=os.path.join(self.output_dir, "final_risk_km.png"), discrete=False)
        
        # 3. Time-dependent ROC
        self.plot_roc_curves(os.path.join(self.output_dir, "final_roc.png"))
        
        # 4. Correlation & VIF
        self.plot_correlation_combo(selected_features_series.index.tolist(), self.output_dir)
        
        # 5. Save Final Markers
        final_df = pd.DataFrame({
            'feature': selected_features_series.index,
            'coef': selected_features_series.values,
            'HR': np.exp(selected_features_series.values)
        })
        final_df.to_csv(os.path.join(self.output_dir, "final_prognostic_biomarkers.csv"), index=False)
        print(f"Final biomarkers saved to {os.path.join(self.output_dir, 'final_prognostic_biomarkers.csv')}")

    def plot_roc_curves(self, output_path):
        time_points = [365, 1095, 1825] # 1, 3, 5 years
        colors = ['blue', 'red', 'green']
        plt.figure(figsize=(8, 8))
        
        for tp, color in zip(time_points, colors):
            subset_mask = (self.df['T'] >= tp) | (self.df['E'] == 1)
            subset = self.df[subset_mask].copy()
            subset['label'] = (subset['T'] < tp) & (subset['E'] == 1)
            subset['label'] = subset['label'].astype(int)
            
            if len(subset['label'].unique()) < 2: continue
                
            fpr, tpr, _ = roc_curve(subset['label'], subset['RiskScore'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, label=f'{tp//365} Year (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Time-dependent ROC')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.close()

    def plot_correlation_combo(self, genes, output_dir):
        if len(genes) < 3: return
        genes = genes[:20] # Limit to top 20
        X = self.df[genes]
        corr_mat = X.corr()
        
        # VIF
        try:
            X_const = add_constant(X)
            vif_vals = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
            vif_data = pd.Series(vif_vals, index=X_const.columns).drop('const', errors='ignore')
        except:
            vif_data = pd.Series(0, index=genes)

        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2)
        
        # Heatmap
        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', ax=ax1)
        ax1.set_title("Correlation Heatmap")
        
        # Radar
        ax2 = fig.add_subplot(gs[1], polar=True)
        labels = vif_data.index.tolist()
        stats = vif_data.values.tolist()
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]; angles += angles[:1]
        ax2.plot(angles, stats, color='red', linewidth=2)
        ax2.fill(angles, stats, color='red', alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels)
        ax2.set_title("VIF (Collinearity)")
        
        plt.savefig(os.path.join(output_dir, "gene_correlation_vif_combo.png"))
        plt.close()

    def run_full_pipeline(self):
        # 1. Univariate
        candidates, _ = self.univariate_cox()
        
        # 2. LASSO
        if candidates:
            selected = self.lasso_cox(candidates)
            
            # 3. Final Model
            if not selected.empty:
                self.build_final_model(selected)
                
                # 4. Plot Top 3 KM
                top_feats = selected.abs().sort_values(ascending=False).head(3).index
                for feat in top_feats:
                    self.plot_km(feat, os.path.join(self.output_dir, f"km_{feat}.png"), discrete=True)
            else:
                print("No features selected by LASSO.")
        else:
            print("No significant features found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="e:\\TERM\\data", help="Path to data directory")
    parser.add_argument("--output_dir", default="e:\\TERM\\results", help="Path to output directory")
    args = parser.parse_args()
    
    try:
        ta = TraditionalAnalysis(args.data_dir, args.output_dir)
        ta.run_full_pipeline()
        print("\nPipeline execution completed successfully.")
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
