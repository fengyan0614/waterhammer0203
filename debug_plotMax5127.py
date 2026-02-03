import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, ifft

# ==========================================
# 1. 批量配置与真值注入
# ==========================================
# 数据路径 (请确认路径无误)
data_dir = r"D:\movefromHP\研究生\新疆项目-超深层页岩油压裂缝网量化评价技术\水锤数据\Max5127原始数据局\新建文件夹"
output_dir = os.path.join(data_dir, "光纤对标_修正报告")
if not os.path.exists(output_dir): os.makedirs(output_dir)

# --- 诊断配置：射孔簇设计深度 ---
stage_configs = {
    "第4段": [6108.5, 6099.5, 6093.5, 6084.5, 6075.5, 6066.5],
    "第5-5段": [6055.0, 6042.5, 6031.0, 6017.5],
    "第11段": [5755.5, 5743.5, 5734.5, 5720.5],
    "第12段": [5703.5, 5690.5, 5681.0, 5664.0],
    "第13段": [5652.5, 5640.5, 5628.5, 5617.5],
    "第14段": [5606.0, 5595.0, 5585.5, 5573.5],
    "第15段": [5561.0, 5541.5, 5528.5],
    "第16段": [5516.0, 5508.0, 5499.0, 5487.5, 5482.0, 5476.0],
    "第17段": [5461.0, 5448.5, 5437.5, 5428.0],
    "第18段": [5414.5, 5404.0, 5393.0, 5380.5],
    "第19段": [5362.5, 5348.5, 5337.0, 5324.5],
    "第20段": [5313.5, 5302.0, 5284.5, 5274.0],
    "第21段": [5258.0, 5247.0, 5235.0, 5223.5],
}

# --- 光纤真值 (Ground Truth) ---
# 1=开启, 0=未开启
fiber_ground_truth = {
    "第4段": [0, 1, 0, 1, 1, 0],
    "第5-5段": [0, 0, 1, 0],
    "第11段": [1, 1, 1, 0],
    "第12段": [1, 0, 1, 1],
    "第13段": [1, 1, 1, 1],
    "第14段": [1, 1, 1, 0],
    "第15段": [0, 0, 1],
    "第16段": [1, 0, 0, 1, 1, 0],
    "第17段": [1, 1, 0, 1],
    "第18段": [1, 1, 1, 0],
    "第19段": [0, 1, 1, 1],
    "第20段": [1, 1, 1, 0],
    "第21段": [0, 1, 0, 1],
}

# 物理与算法参数
# 搜索范围：在 1250 m/s 到 1480 m/s 之间寻找最佳波速
speed_search_range = np.arange(1250, 1480, 2.0)
filter_low, filter_high = 2.0, 45.0
clip_time = 70
current_threshold = 0.0010  # 稍微降低阈值，避免漏掉弱信号
match_error_limit = 6.0  # 允许误差范围 (m)
que_buffer = 0.2

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 核心处理逻辑
# ==========================================

def get_signal_data(file_path):
    """读取并处理信号，返回倒频谱"""
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"读取失败: {file_path}")
        return None, None, None

    time, pressure = df.iloc[:, 0].values, df.iloc[:, 1].values
    if clip_time > 0 and time[-1] > clip_time:
        idx = np.argmax(time > clip_time)
        time, pressure = time[:idx], pressure[:idx]

    dt = np.median(np.diff(time))
    fs = 1.0 / dt if dt > 0 else 1000.0

    # 滤波预处理
    p_med = np.median(pressure)
    p_mad = np.median(np.abs(pressure - p_med))
    pressure = np.clip(pressure, p_med - 3 * p_mad, p_med + 3 * p_mad)

    nyq = 0.5 * fs
    b, a = butter(5, [filter_low / nyq, filter_high / nyq], btype='band')
    p_filt = filtfilt(b, a, pressure)

    spec = fft(p_filt)
    # 使用 log(abs) 计算倒谱，这里依然保留 abs，因为我们主要看能量分布
    log_spec = np.log(np.abs(spec) / np.max(np.abs(spec)) + 1e-10)
    cep = np.abs(ifft(log_spec))
    que = np.arange(len(cep)) / fs
    return que, cep, dt


def calculate_match_score(que, cep, target_depths, fiber_truth, test_speed):
    """
    计算特定波速下的匹配分数
    分数规则：
    1. 击中光纤为1的簇 -> +10分
    2. 击中光纤为0的簇 -> -5分 (惩罚误报)
    3. 总误差越小分数越高
    """
    # 提取当前波速下的所有潜在峰值
    t_min = (2 * min(target_depths)) / test_speed - 0.2
    t_max = (2 * max(target_depths)) / test_speed + 0.2
    mask = (que > t_min) & (que < t_max)
    v_que, v_cep = que[mask], cep[mask]

    if len(v_que) == 0: return -999, [], []

    peaks, _ = find_peaks(v_cep, prominence=current_threshold)
    detected_depths = (test_speed * v_que[peaks]) / 2

    score = 0
    matched_status = ['×'] * len(target_depths)

    # 对每个设计深度，寻找最近的检测峰
    for i, target in enumerate(target_depths):
        expected_truth = fiber_truth[i]  # 1 or 0

        # 找最近的峰
        if len(detected_depths) > 0:
            errors = np.abs(detected_depths - target)
            min_err_idx = np.argmin(errors)
            min_err = errors[min_err_idx]

            if min_err < match_error_limit:
                # 找到了匹配信号
                matched_status[i] = '√'
                if expected_truth == 1:
                    score += 10  # 正确识别开启
                    score -= min_err * 0.1  # 误差惩罚
                else:
                    score -= 5  # 错误识别（光纤说没开，水锤说开了）
            else:
                # 没找到信号
                if expected_truth == 1:
                    score -= 5  # 漏报（光纤说开了，水锤没找到）
                else:
                    score += 2  # 正确排除（都没开）

    return score, matched_status, detected_depths


def auto_optimize_speed(que, cep, target_depths, fiber_truth):
    """遍历波速范围，寻找与光纤结果最吻合的波速"""
    best_speed = 1350.0
    best_score = -9999

    for v in speed_search_range:
        score, _, _ = calculate_match_score(que, cep, target_depths, fiber_truth, v)
        if score > best_score:
            best_score = score
            best_speed = v

    return best_speed, best_score


def final_analysis(que, cep, target_depths, fiber_truth, optimal_speed):
    """基于最优波速输出最终结果"""
    # 这里复用之前的逻辑，但是增加了光纤列
    t_min = (2 * min(target_depths)) / optimal_speed - que_buffer
    t_max = (2 * max(target_depths)) / optimal_speed + que_buffer
    mask = (que > t_min) & (que < t_max)
    v_que, v_cep = que[mask], cep[mask]

    peaks, _ = find_peaks(v_cep, prominence=current_threshold)
    detected_p = [{'depth': (optimal_speed * v_que[p]) / 2, 'amp': v_cep[p], 'que': v_que[p]} for p in peaks]

    res = []
    for i, target in enumerate(target_depths):
        res.append({
            'cluster': f"第{i + 1}簇",
            'target': target,
            'fiber': fiber_truth[i],  # 光纤真值
            'wh_status': 0,  # 水锤推断 (0/1)
            'depth': "--", 'error': "--", 'que': None, 'amp': 0
        })

    # 匹配逻辑
    for i, target in enumerate(target_depths):
        best_match = None
        min_err = match_error_limit

        for p_data in detected_p:
            err = abs(p_data['depth'] - target)
            if err < min_err:
                min_err = err
                best_match = p_data

        if best_match:
            res[i].update({
                'wh_status': 1,
                'depth': round(best_match['depth'], 2),
                'error': round(best_match['depth'] - target, 2),
                'que': best_match['que'], 'amp': best_match['amp']
            })

    return res, (v_que, v_cep)


# ==========================================
# 3. 执行引擎
# ==========================================
if __name__ == "__main__":
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    print(f"{'段号':<8} | {'最优波速':<10} | {'光纤一致率':<10}")
    print("-" * 40)

    for stage_name, targets in stage_configs.items():
        # 1. 匹配文件
        f_path = next((f for f in all_files if stage_name in f), None)
        if not f_path: continue

        # 2. 获取真值
        truth = fiber_ground_truth.get(stage_name)
        if not truth:
            print(f"⚠️ {stage_name} 缺光纤数据，跳过")
            continue

        # 3. 数据读取
        que, cep, _ = get_signal_data(f_path)
        if que is None: continue

        # 4. 核心：基于真值反演最优波速
        best_speed, best_score = auto_optimize_speed(que, cep, targets, truth)

        # 5. 生成最终结果
        results, (v_que, v_cep) = final_analysis(que, cep, targets, truth, best_speed)

        # 计算一致性 (Acc)
        match_count = sum(1 for r in results if r['fiber'] == r['wh_status'])
        acc = (match_count / len(targets)) * 100
        print(f"{stage_name:<8} | {best_speed:.1f} m/s  | {acc:.1f}%")

        # --- 绘图逻辑优化 ---
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.8, 1], width_ratios=[3, 1], hspace=0.35)

        # A. 倒频谱图
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(v_que, v_cep, color='#1f77b4', alpha=0.6, lw=1.5, label='倒谱信号')

        # 绘制靶点
        for r in results:
            t_theo = (2 * r['target']) / best_speed
            # 颜色逻辑：光纤说开了是绿色，没开是灰色
            line_color = 'green' if r['fiber'] == 1 else 'gray'
            line_style = '--' if r['fiber'] == 1 else ':'
            alpha_val = 0.8 if r['fiber'] == 1 else 0.4

            ax0.axvline(x=t_theo, color=line_color, ls=line_style, alpha=alpha_val)
            ax0.text(t_theo, ax0.get_ylim()[1] * 0.95,
                     f"{r['cluster']}\n(光纤:{r['fiber']})",
                     rotation=90, color=line_color, ha='right', fontsize=10)

            if r['wh_status'] == 1:
                # 如果水锤也检测到了，画个红点
                ax0.plot(r['que'], r['amp'], "ro", ms=8)

        ax0.set_title(f"【{stage_name}】光纤对标诊断 (反演波速: {best_speed:.1f} m/s | 一致率: {acc:.0f}%)",
                      fontsize=14, fontweight='bold')
        ax0.set_xlabel("倒频率 (s)", fontsize=12)
        ax0.set_ylabel("幅值", fontsize=12)
        ax0.legend(loc='upper right')

        # B. 对比柱状图 (左：光纤，右：水锤)
        ax1 = fig.add_subplot(gs[0, 1])
        y_pos = np.arange(len(targets))
        bar_width = 0.35

        fiber_vals = [r['fiber'] for r in results]
        wh_vals = [r['wh_status'] for r in results]

        # 绘制
        ax1.barh(y_pos + bar_width / 2, fiber_vals, bar_width, label='光纤(真值)', color='#2ca02c', alpha=0.7)
        ax1.barh(y_pos - bar_width / 2, wh_vals, bar_width, label='水锤(诊断)', color='#d62728', alpha=0.7)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([r['cluster'] for r in results])
        ax1.set_xlim(0, 1.2)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['闭合', '开启'])
        ax1.invert_yaxis()  # 簇1在最上面
        ax1.legend()
        ax1.set_title("开启一致性对比")

        # C. 详细表格
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        table_vals = []
        for r in results:
            # 状态判定
            if r['fiber'] == 1 and r['wh_status'] == 1:
                eval_str = "准确(TP)"
            elif r['fiber'] == 0 and r['wh_status'] == 0:
                eval_str = "准确(TN)"
            elif r['fiber'] == 1 and r['wh_status'] == 0:
                eval_str = "漏报(FN)"
            else:
                eval_str = "误报(FP)"

            # 颜色高亮
            row_color = "#e6f5e6" if "准确" in eval_str else "#ffe6e6"

            table_vals.append([
                r['cluster'],
                f"{r['target']}m",
                "开启" if r['fiber'] == 1 else "闭合",
                "开启" if r['wh_status'] == 1 else "闭合",
                eval_str,
                f"{r['error']}m" if r['error'] != "--" else "--"
            ])

        col_labels = ["簇号", "设计深度", "光纤结果", "水锤结果", "对标评价", "定位误差"]
        diag_table = ax_table.table(cellText=table_vals, colLabels=col_labels,
                                    loc='center', cellLoc='center', bbox=[0, 0.1, 1, 0.9])
        diag_table.auto_set_font_size(False)
        diag_table.set_fontsize(12)

        # 保存
        plt.savefig(os.path.join(output_dir, f"{stage_name}_光纤对标分析.png"), dpi=200, bbox_inches='tight')
        plt.close()  # 关闭图像释放内存

    print(f"\n✨ 分析完成！请查看目录: {output_dir}")
