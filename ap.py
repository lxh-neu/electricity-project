#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终版 - 时间序列电力客户聚类分析程序 (PCA + Affinity Propagation)
- 核心方法:
  - 使用PCA进行特征降维，保留95%的方差。
  - 使用 Affinity Propagation (AP) 算法进行聚类，能自动发现聚类数量。
  - AP的聚类中心是真实的数据点（“exemplars”），结果易于解读。
- 运行环境:
  - 纯CPU实现 (Pandas, Scikit-learn)。
- 注意:
  - Affinity Propagation 算法复杂度高 (O(N^2))，在超大数据集上可能运行缓慢。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime, timedelta
import logging
import xlwings as xw
import random
from collections import defaultdict

# Scikit-learn for CPU-based ML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

# --- 配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class AffinityPropagationClusterer:
    """电力客户用电模式分类器 (PCA + Affinity Propagation)"""
    
    def __init__(self, data_path='2024'):
        self.data_path = data_path
        self.raw_15min_data = {}
        self.processed_data = {}
        self.features = {}
        self.clusters = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95) # 目标：保留95%的方差

    def _clear_excel_styles_xlwings(self, file_path, output_path):
        """【已修复】使用xlwings清除Excel样式，通过显式创建App实例提高稳定性。"""
        try:
            with xw.App(visible=False, add_book=False) as app:
                wb = app.books.open(os.path.abspath(file_path))
                for sheet in wb.sheets:
                    if sheet.used_range and sheet.used_range.count > 1:
                        sheet.used_range.clear_formats()
                wb.save(os.path.abspath(output_path))
                wb.close()
            logger.info(f"样式已清除，文件保存为: {output_path}")
            return True
        except Exception as e:
            logger.error(f"使用xlwings清除样式失败: {str(e)}")
            try:
                for app_instance in xw.apps: app_instance.quit()
            except Exception as cleanup_error:
                logger.debug(f"清理Excel进程时发生额外错误 (可忽略): {cleanup_error}")
            return False

    def load_raw_data(self):
        # (此函数无需修改，使用Pandas)
        customer_folders = sorted([f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))])
        logger.info(f"发现 {len(customer_folders)} 个客户文件夹，开始加载数据...")
        for customer_id in customer_folders:
            customer_path = os.path.join(self.data_path, customer_id)
            power_file = next((os.path.join(customer_path, f) for f in os.listdir(customer_path) if f.startswith('有功功率') and f.endswith('.xlsx')), None)
            if not power_file: continue
            cleaned_file = os.path.join(customer_path, '有功功率_清理.xlsx')
            try:
                if os.path.exists(cleaned_file): df = pd.read_excel(cleaned_file)
                else:
                    try: df = pd.read_excel(power_file)
                    except Exception:
                        if self._clear_excel_styles_xlwings(power_file, cleaned_file): df = pd.read_excel(cleaned_file)
                        else: continue
                time_col = next((c for c in df.columns if '時間' in c or '时间' in c or 'time' in c.lower()), None)
                power_col = next((c for c in df.columns if '有功功率' in c or '功率' in c or 'power' in c.lower()), None)
                if not time_col or not power_col: continue
                processed_df = df[[time_col, power_col]].copy()
                processed_df.columns = ['时间', '功率']
                processed_df['时间'] = pd.to_datetime(processed_df['时间'], errors='coerce')
                processed_df['功率'] = pd.to_numeric(processed_df['功率'], errors='coerce')
                processed_df = processed_df.dropna(subset=['时间']).set_index('时间').sort_index()
                full_index = pd.date_range(start=processed_df.index.min(), end=processed_df.index.max(), freq='15min')
                self.raw_15min_data[customer_id] = processed_df.reindex(full_index)
            except Exception as e: logger.error(f"处理客户 {customer_id} 数据时发生意外错误: {str(e)}")
        logger.info(f"总共成功加载 {len(self.raw_15min_data)} 个客户的原始数据。")

    def impute_missing_values(self, df, customer_id):
        # (此函数无需修改，使用Pandas)
        df = df.copy()
        missing_count = df['功率'].isnull().sum()
        if missing_count == 0: return df
        df['time_of_day'] = df.index.time
        missing_indices = df[df['功率'].isnull()].index
        for missing_time in missing_indices:
            same_time_values = []
            for days_offset in range(-7, 8):
                if days_offset == 0: continue
                target_datetime = missing_time + timedelta(days=days_offset)
                if target_datetime in df.index and not pd.isna(df.loc[target_datetime, '功率']):
                    same_time_values.append(df.loc[target_datetime, '功率'])
            if same_time_values: df.loc[missing_time, '功率'] = np.mean(same_time_values)
        df['功率'] = df['功率'].interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        return df.drop(columns=['time_of_day'])

    def prepare_data_for_analysis(self, analysis_type):
        # (此函数无需修改，使用Pandas)
        self.processed_data = {}
        logger.info(f"--- 为 [{analysis_type}] 分析准备数据 ---")
        for customer_id, df in self.raw_15min_data.items():
            summer_data = df[df.index.month.isin([6, 7, 8, 9])]
            if summer_data.empty: continue
            if analysis_type == '工作日': period_data = summer_data[summer_data.index.dayofweek.isin(range(5))]
            else: period_data = summer_data[summer_data.index.dayofweek.isin(range(5, 7))]
            if period_data.empty: continue
            imputed_data = self.impute_missing_values(period_data, customer_id)
            hourly_data = imputed_data.resample('H').mean().dropna()
            if hourly_data.empty: continue
            hourly_data['小时'] = hourly_data.index.hour
            self.processed_data[customer_id] = hourly_data
        logger.info(f"[{analysis_type}] 数据准备完成，共有 {len(self.processed_data)} 个客户用于分析。")

    def extract_features(self, analysis_type):
        # (此函数无需修改)
        self.features = {}
        for customer_id, df in self.processed_data.items():
            if df.empty: continue
            features = {}
            daily_profile = df.groupby('小时')['功率'].mean()
            for i in range(24): features[f'小时_{i:02d}'] = daily_profile.get(i, 0)
            for peak_name, (start, end) in {'早高峰': (7, 10), '中高峰': (11, 14), '晚高峰': (17, 21)}.items():
                peak_df = df[df['小时'].between(start, end, inclusive='both')]
                features[f'{peak_name}平均功率'] = peak_df['功率'].mean() if not peak_df.empty else 0
                features[f'{peak_name}总用电量'] = peak_df['功率'].sum() if not peak_df.empty else 0
            max_power, min_power, avg_power = df['功率'].max(), df['功率'].min(), df['功率'].mean()
            features['峰谷比'] = max_power / (min_power + 1e-9)
            features['负荷率'] = avg_power / (max_power + 1e-9)
            self.features[customer_id] = {k: v if not pd.isna(v) else 0 for k, v in features.items()}
        logger.info(f"[{analysis_type}] 特征提取完成。")

    # ==============================================================================
    # ===== 核心修改 1：全新的聚类函数 (PCA + Affinity Propagation) =====
    # ==============================================================================
    def run_affinity_propagation_clustering(self, analysis_type, damping, preference):
        """执行PCA降维和Affinity Propagation聚类"""
        if not self.features:
            logger.error(f"[{analysis_type}] 没有特征数据，无法聚类。")
            return False
        
        feature_df = pd.DataFrame.from_dict(self.features, orient='index').fillna(0)
        
        logger.info("步骤 1/3: 标准化特征...")
        scaled_features = self.scaler.fit_transform(feature_df)
        
        logger.info("步骤 2/3: PCA降维 (保留95%方差)...")
        features_pca = self.pca.fit_transform(scaled_features)
        logger.info(f"PCA完成，数据从 {scaled_features.shape[1]} 维降至 {features_pca.shape[1]} 维。")
        
        logger.info(f"步骤 3/3: 执行Affinity Propagation (damping={damping}, preference={preference})...")
        logger.warning("注意：此步骤对于大数据集可能需要较长时间！")
        ap = AffinityPropagation(damping=damping, preference=preference, random_state=42)
        ap.fit(features_pca)
        
        labels = ap.labels_
        exemplar_indices = ap.cluster_centers_indices_
        n_clusters = len(exemplar_indices)
        
        if n_clusters == 0:
            logger.error("Affinity Propagation 未能找到任何聚类，请尝试调整 'preference' 参数。")
            return False
        
        exemplar_customer_ids = feature_df.index[exemplar_indices].tolist()
        logger.info(f"AP聚类完成，发现 {n_clusters} 个聚类。")
        logger.info(f"代表性客户 (Exemplars) ID: {exemplar_customer_ids}")
        
        silhouette_avg = silhouette_score(features_pca, labels) if n_clusters > 1 else 0
        if n_clusters > 1:
            logger.info(f"轮廓系数: {silhouette_avg:.4f}")

        self.clusters = {
            'customer_ids': feature_df.index.tolist(),
            'labels': labels,
            'features_pca': features_pca,
            'exemplar_ids': exemplar_customer_ids,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg
        }
        return True

    def analyze_clusters(self):
        # (此函数逻辑与K-Means版本类似，无需处理噪声点)
        cluster_analysis = {}
        customer_ids, labels = self.clusters['customer_ids'], self.clusters['labels']
        
        for cluster_id in sorted(np.unique(labels)):
            customer_mask = (labels == cluster_id)
            cluster_customers = [customer_ids[i] for i, is_in in enumerate(customer_mask) if is_in]
            
            cluster_feature_df = pd.DataFrame.from_dict({c: self.features[c] for c in cluster_customers}, orient='index')
            avg_features = cluster_feature_df.mean().to_dict()
            
            peaks = {'早高峰型': avg_features.get('早高峰平均功率', 0), '中高峰型': avg_features.get('中高峰平均功率', 0), '晚高峰型': avg_features.get('晚高峰平均功率', 0)}
            peak_type = max(peaks, key=peaks.get) if any(v > 0 for v in peaks.values()) else "平坦型"
                
            cluster_analysis[cluster_id] = {'customers': cluster_customers, 'count': len(cluster_customers), 'peak_type': peak_type, 'features': avg_features}
        return cluster_analysis

    def visualize_and_save_results(self, cluster_analysis, analysis_type, damping, preference):
        # (可视化函数已适配AP结果，无噪声点处理)
        try:
            param_str = f"damp{damping}_pref{preference}"
            plt.switch_backend('Agg')
            fig, axes = plt.subplots(2, 2, figsize=(16, 13))

            counts = [info['count'] for info in cluster_analysis.values()]
            labels_pie = [f"聚类{i} ({info['peak_type']})\n{info['count']}个" for i, info in cluster_analysis.items()]
            axes[0, 0].pie(counts, labels=labels_pie, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title(f'客户聚类分布', fontweight='bold')

            for cid, info in cluster_analysis.items():
                axes[0, 1].plot(range(24), [info['features'].get(f'小时_{i:02d}', 0) for i in range(24)], marker='o', label=f"聚类{cid} ({info['peak_type']})")
            axes[0, 1].set_title('各聚类日平均负荷曲线'); axes[0, 1].legend()

            peak_features = ['早高峰平均功率', '中高峰平均功率', '晚高峰平均功率']
            n_clusters = len(cluster_analysis)
            x, width = np.arange(len(peak_features)), 0.8 / n_clusters
            for i, (cid, info) in enumerate(cluster_analysis.items()):
                offset = width * (i - (n_clusters - 1) / 2)
                axes[1, 0].bar(x + offset, [info['features'][feat] for feat in peak_features], width, label=f"聚类{cid}")
            axes[1, 0].set_title('各聚类高峰期功率对比'); axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(['早高峰', '中高峰', '晚高峰']); axes[1, 0].legend()

            labels, features_pca = self.clusters['labels'], self.clusters['features_pca']
            unique_labels = sorted(set(labels))
            colors = plt.cm.get_cmap('tab10', n_clusters)
            for i, cid in enumerate(unique_labels):
                mask = (labels == cid)
                axes[1, 1].scatter(features_pca[mask, 0], features_pca[mask, 1], color=colors(i), label=f"聚类{cid}")
            axes[1, 1].set_title(f'PCA降维可视化'); axes[1, 1].legend()
            
            fig.suptitle(f'{analysis_type} 分析 (damping={damping}, preference={preference})', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'AP_{analysis_type}_{param_str}_结果.png', dpi=200)
            plt.close()
            
            # 保存CSV
            pd.DataFrame([{'客户ID': cust_id, '聚类': cid, '高峰期类型': info['peak_type']} for cid, info in cluster_analysis.items() for cust_id in info['customers']]).to_csv(f'AP_{analysis_type}_{param_str}_客户结果.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame([{'聚类': cid, **info} for cid, info in cluster_analysis.items()]).to_csv(f'AP_{analysis_type}_{param_str}_特征分析.csv', index=False, encoding='utf-8-sig')

        except Exception as e: logger.error(f"可视化或保存失败: {e}")

    # ==============================================================================
    # ===== 核心修改 2：全新的主流程，测试不同的Affinity Propagation参数 =====
    # ==============================================================================
    def run_full_pipeline(self, ap_params_grid, analysis_types=['工作日', '周末']):
        """运行完整的分析流程，测试多组AP参数"""
        logger.info("="*60); logger.info("开始 Affinity Propagation 聚类分析流程"); self.load_raw_data()
        if not self.raw_15min_data: return
        all_results_summary = []
        for analysis_type in analysis_types:
            logger.info(f"\n{'='*20} 开始 [{analysis_type}] 数据分析 {'='*20}")
            self.prepare_data_for_analysis(analysis_type)
            if not self.processed_data: continue
            self.extract_features(analysis_type)
            if not self.features: continue
            for params in ap_params_grid:
                damping = params.get('damping', 0.9)
                preference = params.get('preference') # Can be None
                logger.info(f"\n--- [{analysis_type}] 分析 (damping={damping}, preference={preference}) ---")
                if not self.run_affinity_propagation_clustering(analysis_type, damping, preference): continue
                cluster_analysis = self.analyze_clusters()
                self.visualize_and_save_results(cluster_analysis, analysis_type, damping, preference)
                summary = {
                    '分析类型': analysis_type, 
                    'damping': damping, 
                    'preference': str(preference), # Convert None to string for CSV
                    '发现的聚类数': self.clusters['n_clusters'],
                    '轮廓系数': self.clusters['silhouette_score']
                }
                all_results_summary.append(summary)
        if all_results_summary:
            summary_df = pd.DataFrame(all_results_summary)
            summary_df.to_csv('AP_综合聚类分析报告.csv', index=False, encoding='utf-8-sig')
            logger.info("\n" + "="*60)
            logger.info("综合分析报告已生成: 'AP_综合聚类分析报告.csv'")
            print("\n综合分析报告摘要：")
            print(summary_df)

def main():
    """主函数入口"""
    clusterer = AffinityPropagationClusterer(data_path='2024')
    
    # 定义要测试的 Affinity Propagation 参数网格
    # preference: 决定聚类数量的关键参数。值越小，聚类数越少。None表示自动估计。
    # damping: 阻尼系数, 0.5 到 1 之间。
    param_grid = [
        {'damping': 0.9, 'preference': -100},   # 推荐的起点：让算法自动估计preference
        {'damping': 0.8, 'preference': -200},  # 尝试用一个负值preference来减少聚类数量
        {'damping': 0.95, 'preference': -300}, # 更大的负值，进一步减少聚类数量
    ]
    
    clusterer.run_full_pipeline(ap_params_grid=param_grid, analysis_types=['工作日', '周末'])

if __name__ == "__main__":
    main()