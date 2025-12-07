import sys
import os

# -----------------------------------------------------------------------------
# Environment Check & Import
# -----------------------------------------------------------------------------
try:
    # Pre-import PyTorch to prevent DLL conflicts with NumPy/Pandas
    import torch
    # print(f"System Check: PyTorch {torch.__version__} initialized successfully.", file=sys.stderr)
except ImportError as e:
    # This will be caught and handled in detail below if critical
    pass
except OSError as e:
    print(f"CRITICAL: PyTorch DLL load failed immediately: {e}", file=sys.stderr)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------------
# CRITICAL: PyTorch Dependency Check & Error Handling
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if float(torch.__version__.split('.')[0]) < 1:
        raise ImportError(f"PyTorch version {torch.__version__} is too old. Please upgrade to >= 1.0.")

except (ImportError, OSError) as e:
    print("\n" + "="*60, file=sys.stderr)
    print("CRITICAL ERROR: PyTorch Initialization Failed", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print("Troubleshooting Guide:", file=sys.stderr)
    print("1. [DLL Error] If seeing WinError 1114/126/127: It's a DLL conflict or missing dependency.", file=sys.stderr)
    print("   - Try reinstalling PyTorch: 'pip install --force-reinstall torch'", file=sys.stderr)
    print("   - Ensure Visual C++ Redistributable is installed.", file=sys.stderr)
    print("2. [Version Error] Ensure PyTorch >= 1.0 is installed.", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
    sys.exit(1)

def _read_tsv(path, index_col=None):
    return pd.read_csv(path, sep="\t", index_col=index_col)

def _find_id_col(df, sample_set):
    best_col = None
    best_score = 0
    for col in df.columns:
        s = set(pd.Series(df[col]).astype(str).dropna().tolist())
        score = len(s & sample_set)
        if score > best_score:
            best_score = score
            best_col = col
    if best_score == 0:
        return None
    return best_col

def merge_data(data_dir="e:\\TERM\\data", missing_threshold=0.2, output_path=None, compute_missing_on="non_expr"):
    expr_path = os.path.join(data_dir, "expression_log2_tpm.tsv")
    clin_path = os.path.join(data_dir, "clinical_cleaned.tsv")
    surv_path = os.path.join(data_dir, "survival_labels.tsv")
    expr = _read_tsv(expr_path, index_col=0).T
    expr.index.name = "sample_id"
    sample_set = set(expr.index.astype(str))
    merged = expr
    expr_cols = set(expr.columns.tolist())
    if os.path.exists(clin_path):
        cdf = _read_tsv(clin_path)
        cid = _find_id_col(cdf, sample_set)
        if cid:
            cdf = cdf.drop_duplicates(subset=[cid]).set_index(cid)
            merged = merged.join(cdf, how="inner")
    if os.path.exists(surv_path):
        sdf = _read_tsv(surv_path)
        sid = _find_id_col(sdf, sample_set)
        if sid:
            sdf = sdf.drop_duplicates(subset=[sid]).set_index(sid)
            merged = merged.join(sdf, how="inner")
    if compute_missing_on == "non_expr":
        cols_for_missing = [c for c in merged.columns if c not in expr_cols]
        if not cols_for_missing:
            cols_for_missing = merged.columns.tolist()
    else:
        cols_for_missing = merged.columns.tolist()
    miss_ratio = merged[cols_for_missing].isna().mean(axis=1)
    merged = merged.loc[miss_ratio <= missing_threshold]
    if output_path is None:
        output_path = os.path.join(data_dir, "merged_dataset.tsv")
    merged.to_csv(output_path, sep="\t", index=True)
    return merged

def _detect_label_cols(df, patterns=None):
    if patterns is None:
        patterns = ("event", "status", "label", "censor", "time")
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if any(p in lc for p in patterns):
            cols.append(c)
            continue
    return cols

def normalize_dataset(input_path="e:\\TERM\\data\\merged_dataset.tsv", output_path=None, exclude_cols=None, label_patterns=None, scale="zscore"):
    df = _read_tsv(input_path, index_col=0)
    if exclude_cols is None:
        exclude_cols = _detect_label_cols(df, patterns=label_patterns)
    num_df = df.select_dtypes(exclude=["object", "category"]).copy()
    cat_df = df.select_dtypes(include=["object", "category"]).copy()
    if exclude_cols:
        num_df_for_scale = num_df.drop(columns=[c for c in exclude_cols if c in num_df.columns], errors="ignore")
    else:
        num_df_for_scale = num_df
    num_df_for_scale = num_df_for_scale.apply(pd.to_numeric, errors="coerce")
    num_df_for_scale = num_df_for_scale.fillna(num_df_for_scale.median())
    if scale == "zscore":
        means = num_df_for_scale.mean()
        stds = num_df_for_scale.std(ddof=0).replace(0, 1)
        num_scaled = (num_df_for_scale - means) / stds
    elif scale == "minmax":
        mins = num_df_for_scale.min()
        maxs = num_df_for_scale.max()
        rng = (maxs - mins).replace(0, 1)
        num_scaled = (num_df_for_scale - mins) / rng
    else:
        num_scaled = num_df_for_scale
    if exclude_cols:
        keep_num = num_df[[c for c in exclude_cols if c in num_df.columns]].copy()
        num_scaled = pd.concat([num_scaled, keep_num], axis=1)
    if not cat_df.empty:
        cat_df = cat_df.apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else ""))
        cat_dum = pd.get_dummies(cat_df, drop_first=False)
    else:
        cat_dum = pd.DataFrame(index=df.index)
    out = pd.concat([num_scaled, cat_dum], axis=1)
    out = out.loc[df.index]
    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_path), "merged_dataset_normalized.tsv")
    out.to_csv(output_path, sep="\t", index=True)
    return out

def qc_by_missing(df, sample_threshold=0.05, feature_threshold=0.05):
    fr = df.isna().mean()
    keep_cols = fr[fr <= feature_threshold].index.tolist()
    df2 = df[keep_cols]
    sr = df2.isna().mean(axis=1)
    df2 = df2.loc[sr <= sample_threshold]
    return df2

def qc_snp(df, maf_threshold=0.01, hwe_threshold=0.01):
    """
    Performs SNP QC: MAF and HWE filtering.
    Expects raw genotype data (0, 1, 2).
    """
    try:
        from scipy.stats import chi2
    except ImportError:
        print("CRITICAL: scipy not installed. Skipping HWE test.")
        return df

    # Identify numeric feature columns (exclude labels)
    excl = _detect_label_cols(df)
    feat_cols = [c for c in df.columns if c not in excl]
    # Select only numeric features for QC
    df_feat = df[feat_cols].select_dtypes(include=[np.number])
    
    # Check if data looks like genotypes (0, 1, 2)
    valid_vals = np.unique(df_feat.values[~np.isnan(df_feat.values)])
    # Allow some float tolerance or strict integers
    if not np.all(np.isin(np.round(valid_vals), [0, 1, 2])):
        print("Warning: Data values do not look like raw genotypes (0/1/2). Skipping SNP QC.")
        return df

    print(f"Starting SNP QC on {df_feat.shape[1]} variants...")

    # 1. MAF Filter
    counts = df_feat.count(axis=0) # Non-NaN count
    sums = df_feat.sum(axis=0)
    freqs = sums / (2 * counts)
    mafs = np.minimum(freqs, 1 - freqs)
    
    keep_maf = mafs >= maf_threshold
    df_maf = df_feat.loc[:, keep_maf]
    dropped_maf = df_feat.shape[1] - df_maf.shape[1]
    print(f"  - MAF Filter (<{maf_threshold}): Dropped {dropped_maf} SNPs")
    
    # 2. HWE Filter
    vals = df_maf.values
    vals = np.round(vals) # Ensure integers
    
    n_0 = np.sum(vals == 0, axis=0)
    n_1 = np.sum(vals == 1, axis=0)
    n_2 = np.sum(vals == 2, axis=0)
    
    n_total = n_0 + n_1 + n_2
    # p = freq of allele 0 (Ref) vs 2 (Alt)
    p = (2 * n_0 + n_1) / (2 * n_total)
    q = 1 - p
    
    e_0 = n_total * (p ** 2)
    e_1 = n_total * (2 * p * q)
    e_2 = n_total * (q ** 2)
    
    e_0 = np.maximum(e_0, 1e-8)
    e_1 = np.maximum(e_1, 1e-8)
    e_2 = np.maximum(e_2, 1e-8)
    
    chisq = ((n_0 - e_0)**2)/e_0 + ((n_1 - e_1)**2)/e_1 + ((n_2 - e_2)**2)/e_2
    p_vals = chi2.sf(chisq, df=1)
    
    keep_hwe = p_vals >= hwe_threshold
    df_final = df_maf.loc[:, keep_hwe]
    dropped_hwe = df_maf.shape[1] - df_final.shape[1]
    print(f"  - HWE Filter (p<{hwe_threshold}): Dropped {dropped_hwe} SNPs")
    
    # Re-attach labels and non-feature cols
    other_cols = df.drop(columns=feat_cols)
    out = pd.concat([df_final, other_cols], axis=1)
    return out

class SoftGAINGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.attn = nn.Linear(dim, dim)  # Attention layer added
        self.fc3 = nn.Linear(dim, dim)
    
    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        
        # Softmax Attention Mechanism with Scaling
        # 使用 Softmax 计算注意力权重，并乘以 dim 进行缩放，防止数值过小导致梯度消失
        attn_weights = F.softmax(self.attn(h), dim=1) * h.shape[1]
        h_weighted = h * attn_weights
        
        # 输出层使用 Sigmoid 确保范围 [0, 1]
        out = torch.sigmoid(self.fc3(h_weighted))
        return out

class SoftGAINDiscriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
    
    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        h_rep = F.relu(self.fc1(inputs))
        h_rep = F.relu(self.fc2(h_rep))
        return torch.sigmoid(self.fc3(h_rep))

def train_soft_gain(data_np, mask_np, epochs=100, batch_size=64, alpha=10.0, beta=1.0):
    dim = data_np.shape[1]
    device = torch.device("cpu") # Use CPU for stability in this env
    
    netG = SoftGAINGenerator(dim).to(device)
    netD = SoftGAINDiscriminator(dim).to(device)
    
    optG = torch.optim.Adam(netG.parameters(), lr=0.001)
    optD = torch.optim.Adam(netD.parameters(), lr=0.001)
    
    data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(data_tensor, mask_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for X_mb, M_mb in loader:
            # Sample noise and hint
            Z_mb = torch.randn_like(X_mb)
            H_mb = M_mb # Simplified hint (can be improved)
            
            # Combine X with Z for missing parts
            X_comb = M_mb * X_mb + (1 - M_mb) * Z_mb
            
            # Train Discriminator
            imputed = netG(X_comb, M_mb)
            X_hat = M_mb * X_mb + (1 - M_mb) * imputed
            
            D_prob = netD(X_hat.detach(), H_mb)
            loss_D = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1 - M_mb) * torch.log(1 - D_prob + 1e-8))
            
            optD.zero_grad()
            loss_D.backward()
            optD.step()
            
            # Train Generator
            imputed = netG(X_comb, M_mb)
            X_hat = M_mb * X_mb + (1 - M_mb) * imputed
            D_prob = netD(X_hat, H_mb)
            
            loss_G_adv = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))
            loss_G_mse = torch.mean((M_mb * X_mb - M_mb * imputed)**2) / torch.mean(M_mb)
            
            # Add Layer-wise penalty (Simplified as L1 sparsity constraint on output for now, 
            # full layer-wise requires accessing intermediate layers)
            loss_G_layer = torch.mean(torch.abs(imputed)) 
            
            loss_G = loss_G_adv + alpha * loss_G_mse + beta * loss_G_layer
            
            optG.zero_grad()
            loss_G.backward()
            optG.step()
            
    # Final Imputation
    with torch.no_grad():
        Z = torch.randn_like(data_tensor)
        X_comb = mask_tensor * data_tensor + (1 - mask_tensor) * Z
        imputed = netG(X_comb, mask_tensor)
        final = mask_tensor * data_tensor + (1 - mask_tensor) * imputed
    
    return final.cpu().numpy()

def impute_with_uncertainty(df, n_runs=5, alpha=10.0, beta=1.0):
    """
    Runs Soft-GAIN multiple times to get mean imputation and uncertainty (std).
    """
    excl = _detect_label_cols(df)
    feat = df.drop(columns=[c for c in excl if c in df.columns], errors="ignore")
    keep = df[[c for c in excl if c in df.columns]].copy()
    
    # Convert to numeric and fill initial NaNs for normalization (GAIN needs normalized input usually)
    # But GAIN handles missingness. We just need to mark NaNs.
    num_df = feat.select_dtypes(exclude=["object", "category"]).copy()
    num_df = num_df.apply(pd.to_numeric, errors="coerce")
    
    data_np = num_df.values
    mask_np = (~np.isnan(data_np)).astype(float)
    # Normalize data to [0, 1] for better GAN training, keeping NaNs
    mins = np.nanmin(data_np, axis=0)
    maxs = np.nanmax(data_np, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    data_norm = (data_np - mins) / ranges
    data_norm = np.nan_to_num(data_norm, nan=0) # Fill NaNs with 0 for input to GAN (masked anyway)
    
    results = []
    for _ in range(n_runs):
        imp = train_soft_gain(data_norm, mask_np, epochs=50, alpha=alpha, beta=beta) # Reduced epochs for speed
        results.append(imp)
    
    results = np.array(results)
    mean_imp = np.mean(results, axis=0)
    std_imp = np.std(results, axis=0)
    
    # Denormalize
    final_imp = mean_imp * ranges + mins
    final_std = std_imp * ranges # Uncertainty in original scale
    
    df_imputed = pd.DataFrame(final_imp, index=num_df.index, columns=num_df.columns)
    df_std = pd.DataFrame(final_std, index=num_df.index, columns=num_df.columns)

    # Re-attach cat cols (mode imputed)
    cat = feat.select_dtypes(include=["object", "category"]).copy()
    if not cat.empty:
        cat = cat.apply(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else ""))
        df_imputed = pd.concat([df_imputed, cat], axis=1)
        # Uncertainty for cat? Ignore for now
        
    # Re-attach labels
    if not keep.empty:
        df_imputed = pd.concat([df_imputed, keep], axis=1)
    
    return df_imputed, df_std

def load_gene_annotation(annot_path=None):
    """
    Loads external annotation to sort genes.
    Expected format: gene_id, chr, start
    """
    if annot_path and os.path.exists(annot_path):
        try:
            annot = pd.read_csv(annot_path, sep="\t")
            # Ensure columns exist
            if {'gene_id', 'chr', 'start'}.issubset(annot.columns):
                # Sort logic: 1-22, X, Y, M
                def chr_key(c):
                    c = str(c).replace("chr", "")
                    if c.isdigit(): return int(c)
                    if c == "X": return 23
                    if c == "Y": return 24
                    if c in ["M", "MT"]: return 25
                    return 99
                
                annot['chr_idx'] = annot['chr'].apply(chr_key)
                annot = annot.sort_values(['chr_idx', 'start'])
                return annot['gene_id'].tolist()
        except Exception as e:
            print(f"Error loading annotation: {e}")
    return None

def reshape_to_2d(df, H=84, W=100, annot_path=None):
    excl = _detect_label_cols(df)
    feat = df.drop(columns=[c for c in excl if c in df.columns], errors="ignore")
    cols = feat.columns.tolist()
    
    # Sort by annotation if available
    sorted_ref = load_gene_annotation(annot_path)
    if sorted_ref:
        # Filter cols that exist in df
        ordered_cols = [c for c in sorted_ref if c in cols]
        # Append remaining cols
        remaining = [c for c in cols if c not in ordered_cols]
        cols_sorted = ordered_cols + sorted(remaining)
    else:
        cols_sorted = sorted(cols, key=lambda x: str(x))
        
    arr = feat[cols_sorted].to_numpy(dtype=np.float32)
    n, d = arr.shape
    m = H * W
    if d < m:
        pad = np.zeros((n, m - d), dtype=np.float32)
        arr2 = np.concatenate([arr, pad], axis=1)
        pad_mask = np.concatenate([np.ones((d,), dtype=np.int32), np.zeros((m - d,), dtype=np.int32)])
    else:
        arr2 = arr[:, :m]
        pad_mask = np.ones((m,), dtype=np.int32)
    matrices = arr2.reshape(n, H, W)
    return cols_sorted, pad_mask, matrices

def preprocess_for_causal(input_path="e:\\TERM\\data\\merged_dataset_normalized.tsv", output_dir=None, H=84, W=100, sample_threshold=0.05, feature_threshold=0.05, annot_path=None, alpha=10.0, beta=1.0, data_type="snp", maf_threshold=0.01, hwe_threshold=0.001):
    df = _read_tsv(input_path, index_col=0)
    
    # 1. QC (Missingness)
    df_qc = qc_by_missing(df, sample_threshold=sample_threshold, feature_threshold=feature_threshold)
    
    # 1.1 QC (SNP Specific)
    if data_type == "snp":
        df_qc = qc_snp(df_qc, maf_threshold=maf_threshold, hwe_threshold=hwe_threshold)
    
    # 2. Soft-GAIN Imputation with Uncertainty
    df_imp, df_std = impute_with_uncertainty(df_qc, n_runs=3, alpha=alpha, beta=beta) # 3 runs default
    
    # 3. Reshape
    cols_sorted, pad_mask, matrices = reshape_to_2d(df_imp, H=H, W=W, annot_path=annot_path)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "causal_preprocess")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save outputs
    pd.Series(cols_sorted).to_csv(os.path.join(output_dir, "feature_order.tsv"), sep="\t", index=False, header=False)
    pd.Series(pad_mask).to_csv(os.path.join(output_dir, "pad_mask.tsv"), sep="\t", index=False, header=False)
    np.save(os.path.join(output_dir, "matrices.npy"), matrices)
    
    # Save uncertainty (flattened or 2D?) -> Let's save as DataFrame for analysis
    df_std.to_csv(os.path.join(output_dir, "imputation_uncertainty.tsv"), sep="\t")
    
    return output_dir

if __name__ == "__main__":
    out_dir = preprocess_for_causal()
    print("saved", out_dir)
