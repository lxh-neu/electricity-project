#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间序列电力客户聚类分析程序 (需求调整版 V3)
- 缺失值处理策略:
  - (保持不变) 使用前后几天同一时刻的平均值填充
- 特征工程:
  - (已更新) 新增了“高峰时段总用电量”作为特征
- 可视化验证:
  - (已修正) 确保样本验证的日期只从夏季（6-9月）中抽取
- 修正:
  - (保持不变) 修复了“关键特征对比”图的显示错误
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import warnings
from datetime import datetime, timedelta
import logging
import xlwings as xw
import random
from collections import defaultdict

# --- 配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class AdjustedElectricPowerClassifierV3:
    """需求调整版电力客户用电模式分类器 (V3)"""
    
    def __init__(self, data_path='2024'):
        self.data_path = data_path
        self.raw_15min_data = {}
        self.processed_data = {}
        self.features = {}
        self.clusters = {}
        self.scaler = StandardScaler()

    def _clear_excel_styles_xlwings(self, file_path, output_path):
        try:
            with xw.App(visible=False) as app:
                wb = app.books.open(os.path.abspath(file_path))
                for sheet in wb.sheets:
                    used_range = sheet.used_range
                    if used_range.count > 1:
                        used_range.clear_formats()
                wb.save(os.path.abspath(output_path))
                wb.close()
            logger.info(f"样式已清除，文件保存为: {output_path}")
            return True
        except Exception as e:
            logger.error(f"使用xlwings清除样式失败: {str(e)}")
            return False

    def load_raw_data(self):
        customer_folders = sorted([f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))])
        logger.info(f"发现 {len(customer_folders)} 个客户文件夹，开始加载数据...")
        
        for customer_id in customer_folders:
            customer_path = os.path.join(self.data_path, customer_id)
            power_file = next((os.path.join(customer_path, f) for f in os.listdir(customer_path) if f.startswith('有功功率') and f.endswith('.xlsx')), None)

            if not power_file:
                logger.warning(f"客户 {customer_id} 没有有功功率Excel文件，跳过。")
                continue
            
            cleaned_file = os.path.join(customer_path, '有功功率_清理.xlsx')
            df = None
            try:
                if os.path.exists(cleaned_file):
                    df = pd.read_excel(cleaned_file)
                else:
                    try:
                        df = pd.read_excel(power_file)
                    except Exception:
                        logger.warning(f"直接读取 {power_file} 失败，尝试使用xlwings清理...")
                        if self._clear_excel_styles_xlwings(power_file, cleaned_file):
                            df = pd.read_excel(cleaned_file)
                        else:
                            logger.error(f"清理失败，跳过客户 {customer_id}。")
                            continue
                
                time_col = next((c for c in df.columns if '時間' in c or '时间' in c or 'time' in c.lower()), None)
                power_col = next((c for c in df.columns if '有功功率' in c or '功率' in c or 'power' in c.lower()), None)

                if not time_col or not power_col:
                    logger.warning(f"客户 {customer_id} 的文件中未找到标准的时间或功率列，跳过。")
                    continue
                
                processed_df = df[[time_col, power_col]].copy()
                processed_df.columns = ['时间', '功率']
                processed_df['时间'] = pd.to_datetime(processed_df['时间'], errors='coerce')
                processed_df['功率'] = pd.to_numeric(processed_df['功率'], errors='coerce')
                processed_df = processed_df.dropna(subset=['时间']).set_index('时间').sort_index()
                
                full_index = pd.date_range(start=processed_df.index.min(), end=processed_df.index.max(), freq='15min')
                processed_df = processed_df.reindex(full_index)

                self.raw_15min_data[customer_id] = processed_df
                logger.info(f"成功加载并预处理客户 {customer_id} 的原始数据，共 {len(processed_df)} 个15分钟数据点。")
                
            except Exception as e:
                logger.error(f"处理客户 {customer_id} 数据时发生意外错误: {str(e)}")
        
        logger.info(f"总共成功加载 {len(self.raw_15min_data)} 个客户的原始数据。")

    def impute_missing_values_reverted(self, df, customer_id):
        df = df.copy()
        missing_count = df['功率'].isnull().sum()
        
        if missing_count == 0:
            return df
            
        logger.info(f"客户 {customer_id} 有 {missing_count} 个缺失值，使用时间窗口方法填充...")
        df['time_of_day'] = df.index.time
        missing_indices = df[df['功率'].isnull()].index
        
        for missing_time in missing_indices:
            current_time_of_day = missing_time.time()
            window_days = 7
            same_time_values = []
            
            for days_offset in range(-window_days, window_days + 1):
                if days_offset == 0: continue
                target_datetime = missing_time + timedelta(days=days_offset)
                if target_datetime in df.index and not pd.isna(df.loc[target_datetime, '功率']):
                    same_time_values.append(df.loc[target_datetime, '功率'])
            
            if same_time_values:
                df.loc[missing_time, '功率'] = np.mean(same_time_values)
            else:
                same_time_data = df[df['time_of_day'] == current_time_of_day]['功率'].dropna()
                if not same_time_data.empty:
                    df.loc[missing_time, '功率'] = same_time_data.mean()
        
        df['功率'] = df['功率'].interpolate(method='time')
        if df['功率'].isnull().any():
            df['功率'] = df['功率'].fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"客户 {customer_id} 使用多种策略填充了 {missing_count} 个缺失值。")
        return df.drop(['time_of_day'], axis=1)

    def prepare_data_for_analysis(self, analysis_type):
        self.processed_data = {}
        logger.info(f"--- 为 [{analysis_type}] 分析准备数据 ---")

        for customer_id, df in self.raw_15min_data.items():
            summer_data = df[df.index.month.isin([6, 7, 8, 9])]
            if summer_data.empty:
                logger.warning(f"客户 {customer_id} 没有夏季月份数据，跳过。")
                continue

            if analysis_type == '工作日':
                period_data = summer_data[summer_data.index.dayofweek.isin(range(5))]
            else:
                period_data = summer_data[summer_data.index.dayofweek.isin(range(5, 7))]

            if period_data.empty:
                logger.warning(f"客户 {customer_id} 在 [{analysis_type}] 没有数据。")
                continue
                
            imputed_data = self.impute_missing_values_reverted(period_data, customer_id)
            hourly_data = imputed_data.resample('H').mean().dropna()

            if hourly_data.empty:
                continue

            hourly_data['小时'] = hourly_data.index.hour
            self.processed_data[customer_id] = hourly_data

        logger.info(f"[{analysis_type}] 数据准备完成，共有 {len(self.processed_data)} 个客户用于分析。")

    # ==============================================================================
    # ===== 修改处 1：extract_features 函数 =====
    # ==============================================================================
    def extract_features(self, analysis_type):
        """特征工程 (已新增高峰时段总用电量)"""
        self.features = {}
        for customer_id, df in self.processed_data.items():
            if df.empty: continue

            features = {}
            # 日平均负荷曲线
            daily_profile = df.groupby('小时')['功率'].mean()
            for i in range(24):
                features[f'小时_{i:02d}'] = daily_profile.get(i, 0)
            
            # 高峰时段平均功率 (反映用电水平)
            features['早高峰平均功率'] = df[df['小时'].between(7, 10)]['功率'].mean() if not df[df['小时'].between(7, 10)].empty else 0
            features['中高峰平均功率'] = df[df['小时'].between(11, 14)]['功率'].mean() if not df[df['小时'].between(11, 14)].empty else 0
            features['晚高峰平均功率'] = df[df['小时'].between(17, 21)]['功率'].mean() if not df[df['小时'].between(17, 21)].empty else 0
            
            # 【新增】高峰时段总用电量 (反映用电总量)
            features['早高峰总用电量'] = df[df['小时'].between(7, 10)]['功率'].sum()
            features['中高峰总用电量'] = df[df['小时'].between(11, 14)]['功率'].sum()
            features['晚高峰总用电量'] = df[df['小时'].between(17, 21)]['功率'].sum()

            # 整体统计特征
            max_power, min_power, avg_power = df['功率'].max(), df['功率'].min(), df['功率'].mean()
            features['峰谷比'] = max_power / (min_power + 1e-6)
            features['负荷率'] = avg_power / (max_power + 1e-6)
            
            self.features[customer_id] = features
        logger.info(f"[{analysis_type}] 特征提取完成，共为 {len(self.features)} 个客户提取了 {len(next(iter(self.features.values())))} 个特征。")

    def perform_clustering(self, n_clusters, analysis_type):
        if not self.features:
            logger.error(f"[{analysis_type}] 没有特征数据，无法聚类。")
            return False
        
        feature_df = pd.DataFrame.from_dict(self.features, orient='index').fillna(0)
        if len(feature_df) < n_clusters:
            logger.error(f"[{analysis_type}] 有效客户数 ({len(feature_df)}) 少于聚类数 ({n_clusters})，无法聚类。")
            return False
            
        scaled_features = self.scaler.fit_transform(feature_df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_features)
        
        score = silhouette_score(scaled_features, labels) if len(set(labels)) > 1 else 0
        
        self.clusters = {
            'customer_ids': feature_df.index.tolist(), 'labels': labels, 'centers': kmeans.cluster_centers_,
            'feature_names': feature_df.columns.tolist(), 'feature_matrix_scaled': scaled_features,
            'silhouette_score': score
        }
        logger.info(f"[{analysis_type}] K-means聚类完成 (K={n_clusters}), 轮廓系数: {score:.4f}")
        return True

    def analyze_clusters(self):
        cluster_analysis = {}
        customer_ids = self.clusters['customer_ids']
        labels = self.clusters['labels']
        
        for cluster_id in sorted(np.unique(labels)):
            customer_mask = (labels == cluster_id)
            cluster_customers = [customer_ids[i] for i, is_in in enumerate(customer_mask) if is_in]
            
            cluster_feature_df = pd.DataFrame.from_dict({c: self.features[c] for c in cluster_customers}, orient='index')
            avg_features = cluster_feature_df.mean().to_dict()
            
            peaks = {
                '早高峰型': avg_features.get('早高峰平均功率', 0),
                '中高峰型': avg_features.get('中高峰平均功率', 0),
                '晚高峰型': avg_features.get('晚高峰平均功率', 0),
            }
            peak_type = max(peaks, key=peaks.get) if any(v > 0 for v in peaks.values()) else "平坦型"

            cluster_analysis[cluster_id] = {
                'customers': cluster_customers, 'count': len(cluster_customers),
                'peak_type': peak_type, 'features': avg_features
            }
        return cluster_analysis
    
    # ==============================================================================
    # ===== 修改处 2：visualize_cluster_samples_96_points 函数 =====
    # ==============================================================================
    def visualize_cluster_samples_96_points(self, cluster_analysis, analysis_type, k_value, samples_per_cluster=3):
        """可视化功能：为每个聚类绘制随机样本的96点日负荷曲线 (已修正日期选择逻辑)"""
        try:
            plt.switch_backend('Agg')
            n_clusters = len(cluster_analysis)
            fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 6 * n_clusters), squeeze=False)
            axes = axes.flatten()

            for cluster_id, info in cluster_analysis.items():
                ax = axes[cluster_id]
                cluster_customers = info['customers']
                n_samples = min(samples_per_cluster, len(cluster_customers))

                if n_samples == 0:
                    ax.text(0.5, 0.5, f'聚类{cluster_id}无数据', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                random.seed(42)
                selected_customers = random.sample(cluster_customers, n_samples)
                
                # 寻找一个所有被选中样本都有数据的共同【夏季】日期
                possible_dates = []
                for customer_id in selected_customers:
                    customer_raw_data = self.raw_15min_data[customer_id]
                    
                    # 【修改】先筛选出夏季的原始数据
                    summer_raw_data = customer_raw_data[customer_raw_data.index.month.isin([6, 7, 8, 9])]
                    
                    if summer_raw_data.empty:
                        possible_dates.append(set()) # 如果该客户无夏季数据，则不可能有共同日期
                        continue

                    # 从夏季数据中根据分析类型（工作日/周末）筛选日期
                    if analysis_type == '工作日':
                        dates = summer_raw_data.index[summer_raw_data.index.dayofweek.isin(range(5))].date
                    else: # 周末
                        dates = summer_raw_data.index[summer_raw_data.index.dayofweek.isin(range(5, 7))].date
                    possible_dates.append(set(np.unique(dates)))
                
                common_dates = list(set.intersection(*possible_dates))
                plot_date = random.choice(common_dates) if common_dates else None

                if not plot_date:
                    ax.text(0.5, 0.5, '无法在夏季为所选样本找到共同的绘图日期', ha='center', va='center', transform=ax.transAxes)
                    logger.warning(f"无法为聚类 {cluster_id} 的样本找到共同的夏季绘图日期。")
                    continue

                for i, customer_id in enumerate(selected_customers):
                    # 从原始数据中提取选定日期的数据进行绘制
                    customer_day_data = self.raw_15min_data[customer_id][self.raw_15min_data[customer_id].index.date == plot_date]
                    if not customer_day_data.empty:
                        ax.plot(range(len(customer_day_data)), customer_day_data['功率'], marker='.', markersize=4,
                                label=f'客户 {customer_id}', alpha=0.8, linewidth=2)
                
                ax.set_title(f'聚类{cluster_id} ({info["peak_type"]}) - K={k_value} - 样本验证 (夏季日期: {plot_date.strftime("%Y-%m-%d")})', fontsize=12, fontweight='bold')
                ax.set_xlabel('每日96个数据点（15分钟间隔）')
                ax.set_ylabel('原始功率 (kW)')
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlim(0, 95)
            
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            output_file = f'{analysis_type}_K{k_value}_聚类样本验证(96点).png'
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            logger.error(f"[{analysis_type}] K={k_value} 96点样本可视化失败: {str(e)}")


    def visualize_and_save_results(self, cluster_analysis, analysis_type, k_value):
        """可视化并保存聚类结果"""
        try:
            plt.switch_backend('Agg')
            fig, axes = plt.subplots(2, 2, figsize=(16, 13))
            
            counts = [info['count'] for info in cluster_analysis.values()]
            labels = [f"聚类{i}\n({info['peak_type']})\n{info['count']}个" for i, info in cluster_analysis.items()]
            axes[0, 0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
            axes[0, 0].set_title(f'客户聚类分布 ({analysis_type}, K={k_value})', fontweight='bold')
            
            time_points = range(24)
            for cid, info in cluster_analysis.items():
                profile = [info['features'][f'小时_{i:02d}'] for i in time_points]
                axes[0, 1].plot(time_points, profile, marker='o', markersize=3, label=f"聚类{cid} ({info['peak_type']})")
            axes[0, 1].set_title('各聚类日平均负荷曲线', fontweight='bold')
            axes[0, 1].set_xlabel('小时')
            axes[0, 1].set_ylabel('平均功率(kW)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            peak_features = ['早高峰平均功率', '中高峰平均功率', '晚高峰平均功率']
            peak_labels = ['早高峰', '中高峰', '晚高峰']
            n_clusters = len(cluster_analysis)
            x = np.arange(len(peak_features))
            width = 0.8 / n_clusters

            for i, (cid, info) in enumerate(cluster_analysis.items()):
                offset = width * (i - (n_clusters - 1) / 2)
                powers = [info['features'][feat] for feat in peak_features]
                axes[1, 0].bar(x + offset, powers, width, label=f"聚类{cid}")

            axes[1, 0].set_title('各聚类高峰期功率对比', fontweight='bold')
            axes[1, 0].set_ylabel('平均功率(kW)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(peak_labels)
            axes[1, 0].legend()
            axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
            
            if self.clusters and len(self.clusters['feature_matrix_scaled']) > 1:
                pca = PCA(n_components=2).fit(self.clusters['feature_matrix_scaled'])
                features_pca = pca.transform(self.clusters['feature_matrix_scaled'])
                labels = self.clusters['labels']
                colors = plt.cm.get_cmap('tab10', k_value)
                for cid in sorted(cluster_analysis.keys()):
                    mask = (labels == cid)
                    axes[1, 1].scatter(features_pca[mask, 0], features_pca[mask, 1], color=colors(cid), label=f"聚类{cid}", alpha=0.7)
                axes[1, 1].set_title(f'PCA降维可视化 (方差解释: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontweight='bold')
                axes[1, 1].set_xlabel('主成分1')
                axes[1, 1].set_ylabel('主成分2')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = f'{analysis_type}_K{k_value}_聚类分析结果.png'
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"聚类分析图已保存至 '{output_file}'")

        except Exception as e:
            logger.error(f"[{analysis_type}] K={k_value} 可视化失败: {str(e)}")

        results_list = []
        for cid, info in cluster_analysis.items():
            for cust_id in info['customers']:
                results_list.append({'客户ID': cust_id, '聚类': cid, '高峰期类型': info['peak_type']})
        pd.DataFrame(results_list).to_csv(f'{analysis_type}_K{k_value}_客户聚类结果.csv', index=False, encoding='utf-8-sig')

        feature_list = []
        for cid, info in cluster_analysis.items():
            row = {'聚类': cid, '高峰期类型': info['peak_type'], '客户数量': info['count']}
            row.update(info['features'])
            row['轮廓系数'] = self.clusters.get('silhouette_score', 0)
            feature_list.append(row)
        pd.DataFrame(feature_list).to_csv(f'{analysis_type}_K{k_value}_聚类特征分析.csv', index=False, encoding='utf-8-sig')
        logger.info(f"CSV结果文件已保存。")


    def run_full_pipeline(self, k_values=[3, 4, 5], analysis_types=['工作日', '周末']):
        logger.info("=" * 60)
        logger.info("开始需求调整版(V3)电力客户聚类分析流程")
        logger.info(f"测试的K值: {k_values}, 分析类型: {analysis_types}")
        logger.info("=" * 60)

        self.load_raw_data()
        if not self.raw_15min_data:
            logger.error("未能加载任何客户数据，程序终止。")
            return None

        all_results_summary = []
        for analysis_type in analysis_types:
            logger.info(f"\n{'='*20} 开始 [{analysis_type}] 数据分析 {'='*20}")
            
            self.prepare_data_for_analysis(analysis_type)
            if not self.processed_data:
                logger.warning(f"[{analysis_type}] 没有有效的客户数据进行分析，跳过。")
                continue
            
            self.extract_features(analysis_type)
            if not self.features:
                logger.error(f"[{analysis_type}] 特征提取失败，跳过。")
                continue
            
            for k in k_values:
                logger.info(f"\n--- [{analysis_type}] K={k} 分析 ---")
                
                if k > len(self.features):
                    logger.warning(f"[{analysis_type}] K={k} 的聚类数大于有效样本数 ({len(self.features)})，跳过此K值。")
                    continue

                if not self.perform_clustering(k, analysis_type):
                    continue
                
                cluster_analysis = self.analyze_clusters()
                self.visualize_and_save_results(cluster_analysis, analysis_type, k)
                self.visualize_cluster_samples_96_points(cluster_analysis, analysis_type, k)

                summary = {'分析类型': analysis_type, 'K值': k, '轮廓系数': self.clusters['silhouette_score']}
                peak_type_counts = defaultdict(int)
                for info in cluster_analysis.values():
                    peak_type_counts[info['peak_type']] += info['count']

                summary['早高峰型客户数'] = peak_type_counts.get('早高峰型', 0)
                summary['中高峰型客户数'] = peak_type_counts.get('中高峰型', 0)
                summary['晚高峰型客户数'] = peak_type_counts.get('晚高峰型', 0)
                summary['总客户数'] = sum(peak_type_counts.values())
                all_results_summary.append(summary)

        if all_results_summary:
            summary_df = pd.DataFrame(all_results_summary).sort_values(by=['分析类型', '轮廓系数'], ascending=[True, False])
            summary_df.to_csv('调整版V3_综合聚类分析报告.csv', index=False, encoding='utf-8-sig')
            logger.info("\n" + "="*60)
            logger.info("综合分析报告已生成: '调整版V3_综合聚类分析报告.csv'")
            print("\n综合分析报告摘要：")
            print(summary_df)
            logger.info("="*60)

        logger.info("\n全部分析流程执行完毕！")


def main():
    classifier = AdjustedElectricPowerClassifierV3(data_path='2024')
    classifier.run_full_pipeline(k_values=[3, 4, 5], analysis_types=['工作日', '周末'])


if __name__ == "__main__":
    main()