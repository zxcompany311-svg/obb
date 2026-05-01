import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import os
import uuid
import zipfile
import imagehash
from pathlib import Path
import requests
import pandas as pd
import openai
import time
import re

st.set_page_config(page_title="电商全能处理中心", layout="wide")

# ==========================================
# 0. 核心配置 (DeepSeek API)
# ==========================================
# 在本地运行时使用硬编码，云端建议使用 st.secrets["DEEPSEEK_API_KEY"]
API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "sk-4f50f524805d4f73ae4893330c2a2b72")
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ==========================================
# 1. 辅助函数
# ==========================================
def save_with_size_limit(img, target_mb, initial_quality=85):
    quality = initial_quality
    img_format = "JPEG" 
    while quality > 10:
        buf = io.BytesIO()
        img.save(buf, format=img_format, quality=quality, subsampling=0)
        size_mb = len(buf.getvalue()) / (1024 * 1024)
        if size_mb <= target_mb or quality <= 20:
            return buf.getvalue(), size_mb, quality
        quality -= 5
    return buf.getvalue(), size_mb, quality

def trim_white_sides(img, threshold):
    data = np.array(img.convert('RGB'))
    gray = np.mean(data, axis=2)
    col_avg = np.mean(gray, axis=0)
    is_white = col_avg >= threshold
    non_white_indices = np.where(~is_white)[0]
    if len(non_white_indices) == 0: return img
    left, right = non_white_indices[0], non_white_indices[-1]
    return img.crop((left, 0, right + 1, img.height))

def contains_chinese(text):
    if not isinstance(text, str): return False
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def calculate_listing_price(cost, domestic_shipping, multiplier, rate, tax):
    try:
        val = float(cost)
        return round((val + domestic_shipping) * rate * multiplier / tax, 1)
    except:
        return 0

def ai_optimize_title(title):
    if not title or pd.isna(title): return ""
    system_prompt = """
    你是一位精通 TikTok Shop 马来西亚站的运营专家。
    任务：将 1688 中文产品标题转化为符合马来西亚市场习惯的 Listing 标题。
    【核心规则】：
    1. 结构：[Ready Stock] + 英语核心词 + 卖点关键词 + | + 马来语核心词 (Bahasa Melayu)。
    2. 严禁出现任何中文字符。
    3. 剔除“2025新款”、“厂家直销”等废话。
    4. 长度：80-120 字符。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"优化标题：{title}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def ai_translate_sku(sku_name):
    if not sku_name or pd.isna(sku_name) or str(sku_name).strip() == "":
        return ""
    if not contains_chinese(str(sku_name)):
        return str(sku_name)
    system_prompt = """
    你是一位跨境电商专家。将 1688 的变体/SKU/属性名称翻译成简洁地道的英语。
    要求：
    1. 仅输出翻译后的英语，不要有任何多余解释。
    2. 严禁出现任何中文字符。
    3. 如果是这种格式：'中文名称-S'，请保留后缀如 '-S' 或 '-XL'，仅翻译前面的中文。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"翻译变体/SKU：{sku_name}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 2. 界面布局
# ==========================================
st.title("📦 电商全能处理中心")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 批量去重模块", 
    "✂️ 裁切/去边/压缩", 
    "💱 汇率换算", 
    "⚖️ 重量换算",
    "🎯 单条 Listing 优化",
    "📦 批量 Listing 处理"
])

# --- TAB 1: 去重 ---
with tab1:
    st.header("1688 采集图内容去重")
    upload_mode = st.radio("选择图片来源：", ["手动上传文件", "直接输入本地文件夹路径"], key="t1_mode")
    files_to_process = []
    if upload_mode == "手动上传文件":
        uploaded_files = st.file_uploader("拖入图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="t1_upload")
        if uploaded_files:
            for f in uploaded_files:
                files_to_process.append({"name": f.name, "data": Image.open(f).convert('RGB')})
    else:
        folder_path = st.text_input("请输入文件夹绝对路径：", key="t1_path")
        if folder_path and os.path.isdir(folder_path):
            p = Path(folder_path)
            all_files = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))
            if st.button("扫描文件夹", key="t1_scan"):
                for f_path in all_files:
                    files_to_process.append({"name": f_path.name, "data": Image.open(f_path).convert('RGB')})
                st.write(f"已找到 {len(files_to_process)} 张图片")
    if files_to_process:
        h_size = st.slider("识别精度", 8, 16, 8, key="t1_hash")
        if st.button("开始清洗重复图片", key="t1_run"):
            seen_hashes, unique_results = set(), []
            bar = st.progress(0)
            for i, item in enumerate(files_to_process):
                h = str(imagehash.average_hash(item["data"], hash_size=h_size))
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_results.append(item)
                bar.progress((i+1)/len(files_to_process))
            st.success(f"清洗完成！保留唯一图 {len(unique_results)}")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as z:
                for res in unique_results:
                    img_buf = io.BytesIO()
                    res["data"].save(img_buf, format="JPEG", quality=95)
                    z.writestr(res["name"], img_buf.getvalue())
            st.download_button("📥 下载去重包", data=zip_buf.getvalue(), file_name="deduplicated.zip")

# --- TAB 2: 裁切 ---
with tab2:
    st.header("Banana2 2K图高级裁切")
    t2_mode = st.radio("选择图片来源：", ["手动上传文件", "直接输入本地文件夹路径"], key="t2_mode")
    t2_files = []
    if t2_mode == "手动上传文件":
        up_files = st.file_uploader("拖入 2048x2048 图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="t2_upload")
        if up_files:
            for f in up_files:
                t2_files.append({"name": f.name, "data": Image.open(f).convert('RGB')})
    else:
        t2_path = st.text_input("请输入文件夹绝对路径：", key="t2_path")
        if t2_path and os.path.isdir(t2_path):
            if st.button("扫描并加载", key="t2_scan"):
                p = Path(t2_path)
                for f_path in (list(p.glob("*.jpg")) + list(p.glob("*.png"))):
                    t2_files.append({"name": f_path.name, "data": Image.open(f_path).convert('RGB')})
    if t2_files:
        col1, col2 = st.columns(2)
        w_thresh = col1.slider("去白边灵敏度", 200, 255, 250)
        mb_limit = col2.number_input("目标体积 (MB)", value=2.0)
        if st.button("执行批量裁切任务", key="t2_run"):
            final_outputs = []
            bar2 = st.progress(0)
            for idx, item in enumerate(t2_files):
                img = item["data"]
                mid = img.width // 2
                parts = [("L", img.crop((0, 0, mid, img.height))), ("R", img.crop((mid, 0, img.width, img.height)))]
                for suffix, p_img in parts:
                    trimmed = trim_white_sides(p_img, w_thresh)
                    f_bytes, _, _ = save_with_size_limit(trimmed, mb_limit)
                    fname = f"{uuid.uuid4().hex[:6]}_{suffix}.jpg"
                    final_outputs.append((fname, f_bytes))
                bar2.progress((idx+1)/len(t2_files))
            st.success(f"裁切完成！共生成 {len(final_outputs)} 张图。")
            zip_buf2 = io.BytesIO()
            with zipfile.ZipFile(zip_buf2, "w") as z:
                for fn, fb in final_outputs: z.writestr(fn, fb)
            st.download_button("📥 下载裁切包", data=zip_buf2.getvalue(), file_name="processed_crops.zip")

# --- TAB 3: 汇率换算 ---
with tab3:
    st.header("💱 跨境汇率极速换算")
    col1, col2, col3 = st.columns([2, 1, 2])
    currencies = {
        "CNY (人民币)": "CNY", "MYR (马来西亚林吉特)": "MYR", "USD (美元)": "USD",
        "IDR (印尼盾)": "IDR", "THB (泰铢)": "THB", "VND (越南盾)": "VND",
        "PHP (菲律宾比索)": "PHP", "SGD (新加坡元)": "SGD"
    }
    with col1:
        base_cur = st.selectbox("从 (原始币种)", list(currencies.keys()), index=0)
        cur_amount = st.number_input("请输入金额", value=100.0, step=0.1, key="cur_amt")
    with col2:
        st.write("")
        st.write("")
        st.markdown("<h2 style='text-align: center;'>⇄</h2>", unsafe_allow_html=True)
    with col3:
        target_cur = st.selectbox("到 (目标币种)", list(currencies.keys()), index=1)
        rate = 1.0
        try:
            r = requests.get(f"https://open.er-api.com/v6/latest/{currencies[base_cur]}")
            if r.status_code == 200:
                rate = r.json()["rates"].get(currencies[target_cur], 1.0)
                st.success(f"实时汇率: 1 {currencies[base_cur]} = {rate:.4f} {currencies[target_cur]}")
        except:
            rate = st.number_input("手动设置汇率", value=0.65, format="%.4f")
    st.metric(label=f"换算结果 ({currencies[target_cur]})", value=f"{cur_amount * rate:,.2f}")

# --- TAB 4: 重量换算 ---
with tab4:
    st.header("⚖️ 重量单位极速换算")
    units = {"克 (g)": 1.0, "千克 (kg)": 1000.0, "磅 (lb)": 453.592, "盎司 (oz)": 28.3495}
    colw1, colw2, colw3 = st.columns([2, 1, 2])
    with colw1:
        from_unit = st.selectbox("从 (原始单位)", list(units.keys()), index=0)
        weight_amt = st.number_input("请输入重量", value=1000.0, step=1.0)
    with colw2:
        st.write("")
        st.write("")
        st.markdown("<h2 style='text-align: center;'>⇄</h2>", unsafe_allow_html=True)
    with colw3:
        to_unit = st.selectbox("到 (目标单位)", list(units.keys()), index=1)
        gram_val = weight_amt * units[from_unit]
        target_val = gram_val / units[to_unit]
    st.metric(label=f"换算结果 ({to_unit})", value=f"{target_val:,.4f}")

# --- TAB 5: 单条商品测试 ---
with tab5:
    st.header("🎯 单条商品 Listing 优化测试")
    st.sidebar.header("⚙️ Listing 定价参数")
    profit_multiplier = st.sidebar.slider("毛利倍数", 1.2, 3.5, 1.8, step=0.1)
    exchange_rate_listing = st.sidebar.number_input("汇率 (1 CNY = ? MYR)", value=0.60)
    domestic_shipping_cny = st.sidebar.number_input("国内运费 (CNY)", value=4.0)
    tax_buffer = st.sidebar.slider("对冲系数 (建议 0.88)", 0.8, 0.95, 0.88)

    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        t_input = st.text_input("1688 原始标题", "2025新款宠物凉感垫夏季降温猫狗窝冰丝垫")
        p_input = st.number_input("成本价格 (CNY)", value=15.0)
        s_input = st.text_input("SKU 规格名 (含中文)", "天空蓝-M码")
    
    if st.button("🚀 开始测试优化"):
        with st.spinner('正在调动 AI...'):
            opt_title = ai_optimize_title(t_input)
            opt_price = calculate_listing_price(p_input, domestic_shipping_cny, profit_multiplier, exchange_rate_listing, tax_buffer)
            opt_sku = ai_translate_sku(s_input)
            with col_l2:
                st.subheader("✨ 优化预览")
                st.metric("建议售价", f"RM {opt_price}")
                st.write("**优化标题:**")
                st.code(opt_title)
                st.write("**SKU 翻译:**")
                st.code(opt_sku)

# --- TAB 6: 批量 Excel 处理 ---
with tab6:
    st.header("📦 批量 Listing 优化 (店小秘/采集箱)")
    uploaded_xlsx = st.file_uploader("上传 Excel 文件", type=["xlsx"], key="batch_xlsx")
    
    if uploaded_xlsx:
        df_raw = pd.read_excel(uploaded_xlsx)
        st.dataframe(df_raw.head(3))
        cols = df_raw.columns.tolist()
        
        t_col = st.selectbox("标题所在的列：", cols, index=0)
        p_col = st.selectbox("采购价所在的列：", cols, index=0)
        s_cols = st.multiselect("需要翻译的 SKU / 属性列：", cols)

        if st.button("⚡ 开始执行全自动化优化"):
            df_work = df_raw.copy()
            progress_bar = st.progress(0)
            status_txt = st.empty()
            
            # 初始化 session_state 存储结果
            if 'processed_df' not in st.session_state:
                st.session_state['processed_df'] = None
                st.session_state['batch_done'] = False

            total = len(df_work)
            for i in range(total):
                status_txt.text(f"⏳ 正在处理第 {i+1}/{total} 条数据...")
                # 标题
                df_work.at[i, t_col] = ai_optimize_title(str(df_work.at[i, t_col]))
                # 定价
                df_work.at[i, p_col] = calculate_listing_price(df_work.at[i, p_col], domestic_shipping_cny, profit_multiplier, exchange_rate_listing, tax_buffer)
                # SKU
                for scol in s_cols:
                    df_work.at[i, scol] = ai_translate_sku(str(df_work.at[i, scol]))
                progress_bar.progress((i + 1) / total)
            
            st.session_state['processed_df'] = df_work
            st.session_state['batch_done'] = True
            st.success("🎉 批量优化成功！")

        if st.session_state.get('batch_done') and st.session_state.get('processed_df') is not None:
            final_df = st.session_state['processed_df']
            
            # 构建 JS 自动填表脚本内容
            mapping = {}
            for i, row in final_df.iterrows():
                old_t = str(df_raw.at[i, t_col]).replace("'", "\\'").replace("\n", " ")
                new_t = str(row[t_col]).replace("'", "\\'").replace("\n", " ")
                mapping[old_t] = new_t
                for scol in s_cols:
                    old_s = str(df_raw.at[i, scol]).replace("'", "\\'").replace("\n", " ")
                    new_s = str(row[scol]).replace("'", "\\'").replace("\n", " ")
                    if old_s != new_s: mapping[old_s] = new_s
            
            import json
            js_mapping = json.dumps(mapping, ensure_ascii=False)
            js_code = f"""(function(){{const m={js_mapping};const ins=document.querySelectorAll('input,textarea');let c=0;ins.forEach(el=>{{const v=el.value?el.value.trim():"";if(m[v]){{el.value=m[v];el.dispatchEvent(new Event('input',{{bubbles:true}}));el.dispatchEvent(new Event('change',{{bubbles:true}}));el.style.backgroundColor='#e1f5fe';c++;}}}});alert('🎉 自动填表完成！共替换了 '+c+' 个位置。');}})();"""
            
            st.subheader("💾 结果下载与自动化")
            st.info("🎯 **店小秘自动填表脚本：** 复制下方代码到 F12 控制台运行。")
            st.code(js_code, language="javascript")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 下载优化后的合规 Excel", data=output.getvalue(), file_name="pawkawan_optimized.xlsx")
