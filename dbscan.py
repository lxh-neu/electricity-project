#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU加速版-高级时间序列电力客户聚类分析程序 (cuDF + cuML) - 最终修复版 V2
- 核心方法:
  - 使用 RAPIDS cuML 在 GPU 上执行 PCA 和 DBSCAN。
- 修正:
  - (已修复) 解决了整数溢出错误 (pyarrow.lib.ArrowInvalid)。
  - (已修复) 移除了所有错误的 super() 调用。
  - (已修复) 解决了 cuDF 无法存储 time 对象的问题。
  - (已修复) 修复了 logging 模块未导入的 NameError。
  - (已修复) 修复了 xlwings 的 'NoneType' 错误。
  - (已修复) 修复了 cuml.metrics.silhouette_score 的导入路径问题。
"""

# CPU-based libraries
import pandas as pd
import numpy as np # <-- 确保 numpy 已导入
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime, timedelta
import logging
import xlwings as xw
import random
from collections import defaultdict

# GPU-based libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
RAPIDS_AVAILABLE = False
print("   将回退到基于 CPU 的 scikit-learn 运行。")


# --- 配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class AdvancedPowerClustererGPU:
    """GPU加速版电力客户用电模式分类器 (最终修复版 V2)"""
    
    def __init__(self, data_path='2024'):
        self.data_path = data_path
        self.raw_15min_data = {}
        self.processed_data = {}
        self.features = {}
        self.clusters = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

    def _clear_excel_styles_xlwings(self, file_path, output_path):
        try:
            with xw.App(visible=False, add_book=False) as app:
                wb = app.books.open(os.path.abspath(file_path))
                for sheet in wb.sheets:
                    used_range = sheet.used_range
                    if used_range and used_range.count > 1:
                        used_range.clear_formats()
                wb.save(os.path.abspath(output_path))
                wb.close()
            logger.info(f"样式已清除，文件保存为: {output_path}")
            return True
        except Exception as e:
            logger.error(f"使用xlwings清除样式失败: {str(e)}")
            return False

    def load_raw_data(self):
        # (此函数无需修改)
        customer_folders = sorted([f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))])
        logger.info(f"发现 {len(customer_folders)} 个客户文件夹，开始加载数据...")
        for customer_id in customer_folders:
            customer_path = os.path.join(self.data_path, customer_id)
            power_file = next((os.path.join(customer_path, f) for f in os.listdir(customer_path) if f.startswith('有功功率') and f.endswith('.xlsx')), None)
            if not power_file: continue
            cleaned_file = os.path.join(customer_path, '有功功率_清理.xlsx')
            df_pd = None
            try:
                if os.path.exists(cleaned_file): df_pd = pd.read_excel(cleaned_file)
                else:
                    try: df_pd = pd.read_excel(power_file)
                    except Exception:
                        if self._clear_excel_styles_xlwings(power_file, cleaned_file): df_pd = pd.read_excel(cleaned_file)
                        else: continue
                time_col = next((c for c in df_pd.columns if '時間' in c or '时间' in c), None)
                power_col = next((c for c in df_pd.columns if '有功功率' in c or '功率' in c), None)
                if not time_col or not power_col: continue
                processed_df_pd = df_pd[[time_col, power_col]].copy()
                processed_df_pd.columns = ['时间', '功率']
                processed_df_pd['时间'] = pd.to_datetime(processed_df_pd['时间'], errors='coerce')
                processed_df_pd['功率'] = pd.to_numeric(processed_df_pd['功率'], errors='coerce')
                processed_df_pd = processed_df_pd.dropna(subset=['时间']).set_index('时间').sort_index()
                if not processed_df_pd.empty:
                    full_index = pd.date_range(start=processed_df_pd.index.min(), end=processed_df_pd.index.max(), freq='15min')
                    processed_df_pd = processed_df_pd.reindex(full_index)
                    if RAPIDS_AVAILABLE: self.raw_15min_data[customer_id] = cudf.from_pandas(processed_df_pd)
                    else: self.raw_15min_data[customer_id] = processed_df_pd
                    logger.info(f"成功加载并预处理客户 {customer_id} 的原始数据。")
            except Exception as e: logger.error(f"处理客户 {customer_id} 数据时发生意外错误: {str(e)}")
        logger.info(f"总共成功加载 {len(self.raw_15min_data)} 个客户的原始数据。")

    def impute_missing_values_reverted(self, df, customer_id):
        # (此函数已修复)
        df_copy = df.copy()
        missing_count_series = df_copy['功率'].isnull().sum()
        missing_count = missing_count_series.item() if hasattr(missing_count_series, 'item') else missing_count_series
        if missing_count == 0: return df_copy
        logger.info(f"客户 {customer_id} 有 {missing_count} 个缺失值，使用时间窗口方法填充...")
        
        if RAPIDS_AVAILABLE:
            # 【核心修复】使用 np.int64 强制64位整数运算，防止溢出
            df_copy['time_numeric'] = (df_copy.index.hour * 3600 + df_copy.index.minute * 60 + df_copy.index.second) * np.int64(1_000_000) + df_copy.index.microsecond
        else:
            df_copy['time_of_day'] = df_copy.index.time

        missing_indices_pd = df_copy['功率'].isna()
        if RAPIDS_AVAILABLE: missing_indices_pd = missing_indices_pd.to_pandas()
        missing_indices = missing_indices_pd[missing_indices_pd].index

        for missing_time in missing_indices:
            window_days, same_time_values_list = 7, []
            for days_offset in range(-window_days, window_days + 1):
                if days_offset == 0: continue
                target_datetime = missing_time + timedelta(days=days_offset)
                if target_datetime in df_copy.index:
                    val = df_copy.loc[target_datetime, '功率']
                    if val is not None and not pd.isna(val):
                        same_time_values_list.append(val)
            if same_time_values_list:
                df_copy.loc[missing_time, '功率'] = np.mean(same_time_values_list)
            else:
                if RAPIDS_AVAILABLE:
                    current_hour = missing_time.hour
                    same_hour_data = df_copy[df_copy.index.hour == current_hour]['功率'].dropna()
                    if not same_hour_data.empty: df_copy.loc[missing_time, '功率'] = same_hour_data.mean().item()
                else:
                    current_time_of_day = missing_time.time()
                    same_time_data = df_copy[df_copy['time_of_day'] == current_time_of_day]['功率'].dropna()
                    if not same_time_data.empty: df_copy.loc[missing_time, '功率'] = same_time_data.mean()
        
        df_copy['功率'] = df_copy['功率'].interpolate()
        if RAPIDS_AVAILABLE:
            df_copy = df_copy.fillna(method='ffill')
            df_copy = df_copy.fillna(method='bfill')
        else:
            df_copy['功率'] = df_copy['功率'].fillna(method='ffill').fillna(method='bfill')
        
        temp_col = 'time_numeric' if RAPIDS_AVAILABLE else 'time_of_day'
        return df_copy.drop(columns=[temp_col])

    def prepare_data_for_analysis(self, analysis_type):
        # (此函数无需修改)
        self.processed_data = {}
        logger.info(f"--- 为 [{analysis_type}] 分析准备数据 ---")
        for customer_id, df in self.raw_15min_data.items():
            summer_data = df[df.index.month.isin([6, 7, 8, 9])]
            if summer_data.empty: continue
            if analysis_type == '工作日':
                period_data = summer_data[summer_data.index.dayofweek.isin(range(5))]
            else:
                period_data = summer_data[summer_data.index.dayofweek.isin(range(5, 7))]
            if period_data.empty: continue
            imputed_data = self.impute_missing_values_reverted(period_data, customer_id)
            hourly_data = imputed_data.resample('H').mean().dropna()
            if hourly_data.empty: continue
            if RAPIDS_AVAILABLE: hourly_data['小时'] = hourly_data.index.hour
            else: hourly_data['小时'] = hourly_data.index.hour
            self.processed_data[customer_id] = hourly_data
        logger.info(f"[{analysis_type}] 数据准备完成，共有 {len(self.processed_data)} 个客户用于分析。")

    def extract_features(self, analysis_type):
        # (此函数无需修改)
        self.features = {}
        for customer_id, df in self.processed_data.items():
            if df.empty: continue
            features, time_ranges = {}, {'早高峰': (7, 10), '中高峰': (11, 14), '晚高峰': (17, 21)}
            daily_profile_gpu = df.groupby('小时')['功率'].mean()
            daily_profile = daily_profile_gpu.to_pandas() if RAPIDS_AVAILABLE else daily_profile_gpu
            for i in range(24): features[f'小时_{i:02d}'] = daily_profile.get(i, 0)
            for peak_name, (start, end) in time_ranges.items():
                if RAPIDS_AVAILABLE:
                    peak_df = df[df['小时'].isin(cp.arange(start, end + 1))]
                else:
                    peak_df = df[df['小时'].between(start, end, inclusive='both')]
                features[f'{peak_name}平均功率'] = peak_df['功率'].mean()
                features[f'{peak_name}总用电量'] = peak_df['功率'].sum()
            max_power, min_power, avg_power = df['功率'].max(), df['功率'].min(), df['功率'].mean()
            features['峰谷比'] = max_power / (min_power + 1e-9)
            features['负荷率'] = avg_power / (max_power + 1e-9)
            self.features[customer_id] = {k: v.item() if hasattr(v, 'item') else (v if not pd.isna(v) else 0) for k, v in features.items()}
        logger.info(f"[{analysis_type}] 特征提取完成。")

    def run_pca_and_dbscan_clustering(self, analysis_type, eps, min_samples):
        # (此函数无需修改)
        if not self.features: return False
        feature_df_pd = pd.DataFrame.from_dict(self.features, orient='index').fillna(0)
        if RAPIDS_AVAILABLE:
            feature_df = cudf.from_pandas(feature_df_pd)
            scaled_features = self.scaler.fit_transform(feature_df)
            features_pca = self.pca.fit_transform(scaled_features)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels_gpu = dbscan.fit_predict(features_pca)
            labels, features_pca_np = labels_gpu.to_numpy(), features_pca.to_numpy()
        else:
            scaled_features = self.scaler.fit_transform(feature_df_pd)
            features_pca = self.pca.fit_transform(scaled_features)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features_pca)
            features_pca_np = features_pca
        n_clusters, n_noise = len(set(labels)) - (1 if -1 in labels else 0), list(labels).count(-1)
        logger.info(f"DBSCAN完成 (eps={eps}, min_samples={min_samples}), 发现 {n_clusters} 个聚类，{n_noise} 个噪声点。")
        if n_clusters > 1:
            if RAPIDS_AVAILABLE:
                score = silhouette_score(features_pca, labels_gpu)
                silhouette_avg = score.item()
            else:
                silhouette_avg = silhouette_score(features_pca, labels)
            logger.info(f"核心点轮廓系数: {silhouette_avg:.4f}")
        else: silhouette_avg = 0
        self.clusters = {'customer_ids': feature_df_pd.index.tolist(), 'labels': labels, 'features_pca': features_pca_np, 'n_clusters': n_clusters, 'n_noise': n_noise, 'silhouette_score': silhouette_avg}
        return True
    
    def analyze_clusters(self):
        # (此函数无需修改)
        cluster_analysis, customer_ids, labels = {}, self.clusters['customer_ids'], self.clusters['labels']
        unique_labels = np.unique(labels)
        for cluster_id in sorted(unique_labels):
            is_noise = (cluster_id == -1)
            customer_mask = (labels == cluster_id)
            cluster_customers = [customer_ids[i] for i, is_in in enumerate(customer_mask) if is_in]
            if is_noise:
                peak_type, avg_features = '噪声/离群点', {}
            else:
                cluster_feature_df = pd.DataFrame.from_dict({c: self.features[c] for c in cluster_customers}, orient='index')
                avg_features = cluster_feature_df.mean().to_dict()
                peaks = {'早高峰型': avg_features.get('早高峰平均功率', 0), '中高峰型': avg_features.get('中高峰平均功率', 0), '晚高峰型': avg_features.get('晚高峰平均功率', 0)}
                peak_type = max(peaks, key=peaks.get) if any(v > 0 for v in peaks.values()) else "平坦型"
            cluster_analysis[cluster_id] = {'customers': cluster_customers, 'count': len(cluster_customers), 'peak_type': peak_type, 'features': avg_features}
        return cluster_analysis
    
    def visualize_and_save_results(self, cluster_analysis, analysis_type, eps, min_samples):
        # (此函数无需修改)
        try:
            plt.switch_backend('Agg')
            fig, axes = plt.subplots(2, 2, figsize=(16, 13))
            core_clusters = {cid: info for cid, info in cluster_analysis.items() if cid != -1}
            counts = [info['count'] for info in cluster_analysis.values()]
            labels_pie = [f"聚类{i} ({info['peak_type']})\n{info['count']}个" if i != -1 else f"{info['peak_type']}\n{info['count']}个" for i, info in cluster_analysis.items()]
            axes[0, 0].pie(counts, labels=labels_pie, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title(f'客户聚类分布', fontweight='bold')
            for cid, info in core_clusters.items():
                axes[0, 1].plot(range(24), [info['features'].get(f'小时_{i:02d}', 0) for i in range(24)], marker='o', label=f"聚类{cid} ({info['peak_type']})")
            axes[0, 1].set_title('各核心聚类日平均负荷曲线'); axes[0, 1].legend()
            peak_features = ['早高峰平均功率', '中高峰平均功率', '晚高峰平均功率']
            if core_clusters:
                x, width = np.arange(len(peak_features)), 0.8 / len(core_clusters)
                for i, (cid, info) in enumerate(core_clusters.items()):
                    offset = width * (i - (len(core_clusters) - 1) / 2)
                    axes[1, 0].bar(x + offset, [info['features'][feat] for feat in peak_features], width, label=f"聚类{cid}")
            axes[1, 0].set_title('各核心聚类高峰期功率对比'); axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(['早高峰', '中高峰', '晚高峰']); axes[1, 0].legend()
            if self.clusters and 'features_pca' in self.clusters:
                labels, features_pca = self.clusters['labels'], self.clusters['features_pca']
                unique_labels = sorted(set(labels))
                colors = plt.cm.get_cmap('tab10', len(unique_labels))
                for i, cid in enumerate(unique_labels):
                    mask = (labels == cid)
                    if cid == -1: axes[1, 1].scatter(features_pca[mask, 0], features_pca[mask, 1], c='gray', marker='x', label='噪声点', s=20)
                    else: axes[1, 1].scatter(features_pca[mask, 0], features_pca[mask, 1], color=colors(i), label=f"聚类{cid}")
                axes[1, 1].set_title(f'PCA降维可视化'); axes[1, 1].legend()
            fig.suptitle(f'{analysis_type} 分析 (eps={eps}, ms={min_samples})', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{analysis_type}_eps{eps}_ms{min_samples}_聚类分析结果.png', dpi=200)
            plt.close()
            pd.DataFrame([{'客户ID': cust_id, '聚类': cid, '高峰期类型': info['peak_type']} for cid, info in cluster_analysis.items() for cust_id in info['customers']]).to_csv(f'{analysis_type}_eps{eps}_ms{min_samples}_客户聚类结果.csv', index=False, encoding='utf-8-sig')
            # 修正保存特征分析的bug
            feature_out_list = []
            for cid, info in core_clusters.items():
                row = {'聚类': cid, '高峰期类型': info['peak_type'], '客户数量': info['count']}
                row.update(info.get('features', {}))
                feature_out_list.append(row)
            pd.DataFrame(feature_out_list).to_csv(f'{analysis_type}_eps{eps}_ms{min_samples}_聚类特征分析.csv', index=False, encoding='utf-8-sig')
        except Exception as e: logger.error(f"可视化或保存失败: {e}")

    def run_full_pipeline(self, dbscan_params_grid, analysis_types=['工作日', '周末']):
        # (此函数无需修改)
        logger.info("="*60); logger.info("开始 GPU 加速版电力客户聚类分析流程"); self.load_raw_data()
        if not self.raw_15min_data: return
        all_results_summary = []
        for analysis_type in analysis_types:
            logger.info(f"\n{'='*20} 开始 [{analysis_type}] 数据分析 {'='*20}")
            self.prepare_data_for_analysis(analysis_type)
            if not self.processed_data: continue
            self.extract_features(analysis_type)
            if not self.features: continue
            for eps, min_samples in dbscan_params_grid:
                logger.info(f"\n--- [{analysis_type}] 分析 (eps={eps}, min_samples={min_samples}) ---")
                if not self.run_pca_and_dbscan_clustering(analysis_type, eps, min_samples): continue
                cluster_analysis = self.analyze_clusters()
                self.visualize_and_save_results(cluster_analysis, analysis_type, eps, min_samples)
                summary = {'分析类型': analysis_type, 'eps': eps, 'min_samples': min_samples, '发现的聚类数': self.clusters['n_clusters'], '噪声点数': self.clusters['n_noise'], '核心点轮廓系数': self.clusters['silhouette_score']}
                all_results_summary.append(summary)
        if all_results_summary:
            summary_df = pd.DataFrame(all_results_summary)
            summary_df.to_csv('GPU版_综合聚类分析报告.csv', index=False, encoding='utf-8-sig')
            print("\n综合分析报告摘要：")
            print(summary_df)

def main():
    classifier = AdvancedPowerClustererGPU(data_path='2024')
    param_grid = [(1.5, 5), (2.0, 5), (2.5, 5)]
    classifier.run_full_pipeline(dbscan_params_grid=param_grid, analysis_types=['工作日', '周末'])

if __name__ == "__main__":
    main()