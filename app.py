# -*- coding: utf-8 -*-
"""
열역학 경로 최적화 시뮬레이터 (Enhanced Version v2.5)
Streamlit 메인 앱

새로운 기능:
- 3D P-V-T 다이어그램
- 다양한 기체 타입 지원
- 열역학 사이클 (오토, 디젤, 브레이턴, 카르노)
- A* 알고리즘 경로 탐색
- 결과 내보내기
- 다크모드
- 한글 폰트 지원
- 향상된 UI/UX
"""

import streamlit as st
import numpy as np
import pandas as pd
from thermodynamics import (
    generate_isothermal_path,
    generate_isobaric_path,
    generate_isochoric_path,
    generate_adiabatic_path,
    calculate_path_properties,
    calculate_temperature,
    generate_otto_cycle,
    generate_diesel_cycle,
    generate_brayton_cycle,
    generate_carnot_cycle,
    GAS_TYPES,
    R, n
)
from pathfinding import find_optimal_path, compare_algorithms
from visualization import (
    plot_pv_diagram,
    plot_work_comparison,
    plot_efficiency_comparison,
    plot_3d_pvt_diagram,
    plot_cycle_diagram,
    plot_thermodynamic_properties,
    plot_algorithm_comparison,
    create_export_data
)

# 페이지 설정
st.set_page_config(
    page_title="열역학 경로 최적화 시뮬레이터",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (향상된 UI/UX v3.0)
st.markdown("""
<style>
    /* 한글 폰트 임포트 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

    /* 전역 폰트 설정 */
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* 메인 헤더 - 네온 글로우 효과 */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6, #00d4ff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 2rem 0 1rem 0;
        animation: neonGlow 8s ease infinite;
        letter-spacing: 2px;
        filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5));
    }

    @keyframes neonGlow {
        0%, 100% { background-position: 0% 50%; filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5)); }
        25% { background-position: 50% 50%; filter: drop-shadow(0 0 25px rgba(124, 58, 237, 0.5)); }
        50% { background-position: 100% 50%; filter: drop-shadow(0 0 30px rgba(244, 114, 182, 0.5)); }
        75% { background-position: 50% 50%; filter: drop-shadow(0 0 25px rgba(124, 58, 237, 0.5)); }
    }

    /* 서브헤더 스타일 */
    .sub-header {
        font-size: 1.2rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 1px;
    }

    /* 글래스모피즘 카드 */
    .glow-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 200, 255, 0.2);
        border-radius: 24px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(20px);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }

    .glow-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }

    .glow-card:hover::before {
        left: 100%;
    }

    .glow-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow:
            0 20px 60px rgba(0, 212, 255, 0.2),
            0 0 40px rgba(124, 58, 237, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.4);
    }

    /* 물리량 그리드 */
    .physics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
    }

    .physics-item {
        text-align: center;
        padding: 12px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        transition: all 0.3s ease;
    }

    .physics-item:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: scale(1.05);
    }

    .physics-symbol {
        font-size: 1.5rem;
        font-weight: 700;
        display: block;
        margin-bottom: 4px;
    }

    .physics-name {
        font-size: 0.85rem;
        color: #a0aec0;
    }

    /* 탭 스타일 - 사이버펑크 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        padding: 10px;
        border-radius: 20px;
        border: 1px solid rgba(100, 200, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 14px 28px;
        border-radius: 14px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #94a3b8;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }

    /* 버튼 스타일 - 홀로그램 효과 */
    .stButton > button {
        border-radius: 14px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(100, 200, 255, 0.2);
        background: rgba(30, 41, 59, 0.8);
        color: #e2e8f0;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.4s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        border-color: rgba(0, 212, 255, 0.5);
        color: #00d4ff;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #7c3aed);
        border: none;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7c3aed, #0ea5e9);
        box-shadow: 0 10px 40px rgba(124, 58, 237, 0.5);
    }

    /* 사이드바 스타일 - 다크 테마 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
        border-right: 1px solid rgba(100, 200, 255, 0.1);
    }

    /* 사이드바 모든 텍스트 밝게 */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        font-weight: 600;
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] input {
        color: #e2e8f0 !important;
        background-color: rgba(30, 41, 59, 0.8) !important;
    }

    /* 익스팬더 스타일 */
    .streamlit-expanderHeader {
        font-weight: 700;
        border-radius: 14px;
        transition: all 0.3s ease;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 200, 255, 0.1);
    }

    .streamlit-expanderHeader:hover {
        background: rgba(0, 212, 255, 0.1);
        border-color: rgba(0, 212, 255, 0.3);
    }

    /* 데이터프레임 스타일 */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(100, 200, 255, 0.1);
    }

    /* 메트릭 위젯 스타일 - 네온 카드 */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(100, 200, 255, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.2);
    }

    [data-testid="stMetricValue"] {
        font-weight: 800;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 500;
    }

    /* 성공/에러/정보 메시지 스타일 */
    .stSuccess {
        border-radius: 14px;
        padding: 1rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .stError {
        border-radius: 14px;
        padding: 1rem;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .stInfo {
        border-radius: 14px;
        padding: 1rem;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .stWarning {
        border-radius: 14px;
        padding: 1rem;
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* 스크롤바 스타일 - 네온 */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0ea5e9, #7c3aed);
        border-radius: 10px;
        border: 2px solid rgba(15, 23, 42, 0.8);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed, #0ea5e9);
    }

    /* 입력 필드 스타일 */
    .stSelectbox > div > div,
    .stSlider > div > div {
        border-radius: 12px;
    }

    /* 동적 배경 */
    .cyber-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        background:
            radial-gradient(ellipse at 20% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(244, 114, 182, 0.03) 0%, transparent 70%);
    }

    /* 그리드 오버레이 */
    .grid-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        background-image:
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
    }

    /* 푸터 스타일 - 퓨처리스틱 */
    .footer {
        text-align: center;
        padding: 3rem 0;
        margin-top: 4rem;
        background: linear-gradient(180deg, transparent, rgba(15, 23, 42, 0.8));
        border-top: 1px solid rgba(100, 200, 255, 0.1);
    }

    .footer-title {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .footer-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .footer-tech {
        color: #475569;
        font-size: 0.8rem;
    }

    /* 펄스 애니메이션 */
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.4); }
        50% { box-shadow: 0 0 0 15px rgba(0, 212, 255, 0); }
    }

    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* 페이드인 애니메이션 */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* 플로팅 애니메이션 */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .floating {
        animation: float 3s ease-in-out infinite;
    }

    /* 상태 배지 스타일 */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 5px;
        transition: all 0.3s ease;
    }

    .status-badge.active {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }

    .status-badge.active:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }

    .status-badge.inactive {
        background: rgba(100, 116, 139, 0.2);
        color: #64748b;
    }

    /* 섹션 타이틀 */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #00d4ff, #7c3aed);
        border-radius: 2px;
    }

    /* 반짝임 효과 */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .shimmer {
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
</style>
<div class="cyber-bg"></div>
<div class="grid-overlay"></div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'paths' not in st.session_state:
    st.session_state.paths = []
if 'optimal_path' not in st.session_state:
    st.session_state.optimal_path = None
if 'P1' not in st.session_state:
    st.session_state.P1 = 5.0
if 'V1' not in st.session_state:
    st.session_state.V1 = 2.0
if 'P2' not in st.session_state:
    st.session_state.P2 = 1.0
if 'V2' not in st.session_state:
    st.session_state.V2 = 8.0
if 'gas_type' not in st.session_state:
    st.session_state.gas_type = 'monatomic'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'cycle_data' not in st.session_state:
    st.session_state.cycle_data = None
# 앱 최초 실행 시 예시 경로 자동 추가
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

# 제목
st.markdown('''
<div class="fade-in" style="text-align: center; padding: 1rem 0;">
    <h1 class="main-header">열역학 경로 최적화 시뮬레이터</h1>
    <p class="sub-header">Thermodynamic Path Optimization Simulator</p>
</div>
''', unsafe_allow_html=True)

# 설명 - 카드 형태로 표시
st.markdown("""
<div class="glow-card fade-in">
    <h4 style="margin-bottom: 1.2rem; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.2rem; font-weight: 700;">
        열역학적 물리량 | Thermodynamic Properties
    </h4>
    <div class="physics-grid">
        <div class="physics-item">
            <span class="physics-symbol" style="color: #00d4ff;">W</span>
            <span class="physics-name">시스템이 한 일</span>
        </div>
        <div class="physics-item">
            <span class="physics-symbol" style="color: #7c3aed;">Q</span>
            <span class="physics-name">시스템이 흡수한 열</span>
        </div>
        <div class="physics-item">
            <span class="physics-symbol" style="color: #f472b6;">ΔU</span>
            <span class="physics-name">내부에너지 변화</span>
        </div>
        <div class="physics-item">
            <span class="physics-symbol" style="color: #fb923c;">ΔH</span>
            <span class="physics-name">엔탈피 변화</span>
        </div>
        <div class="physics-item">
            <span class="physics-symbol" style="color: #4ade80;">ΔS</span>
            <span class="physics-name">엔트로피 변화</span>
        </div>
        <div class="physics-item">
            <span class="physics-symbol" style="color: #22d3d1;">ΔG</span>
            <span class="physics-name">깁스 자유에너지</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.title("⚙️ 설정")

    # 다크모드 토글
    st.session_state.dark_mode = st.toggle("🌙 다크모드", value=st.session_state.dark_mode)

    st.divider()

    # 기체 타입 선택
    st.subheader("🧪 기체 타입")
    gas_options = {
        'monatomic': '단원자 (He, Ne, Ar)',
        'diatomic': '이원자 (N₂, O₂, H₂)',
        'polyatomic': '다원자 (CO₂, H₂O)'
    }
    st.session_state.gas_type = st.selectbox(
        "기체 선택",
        options=list(gas_options.keys()),
        format_func=lambda x: gas_options[x]
    )

    gas_props = GAS_TYPES[st.session_state.gas_type]
    st.info(f"γ = {gas_props['gamma']:.3f}\nCv = {gas_props['Cv']:.4f} L·atm/(mol·K)")

    st.divider()

    # 상태 설정
    st.subheader("📊 상태 설정")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**초기 상태 A**")
        st.session_state.P1 = st.slider("P₁ (atm)", 1.0, 10.0, st.session_state.P1, 0.1)
        st.session_state.V1 = st.slider("V₁ (L)", 1.0, 10.0, st.session_state.V1, 0.1)

    with col_b:
        st.markdown("**최종 상태 B**")
        st.session_state.P2 = st.slider("P₂ (atm)", 1.0, 10.0, st.session_state.P2, 0.1)
        st.session_state.V2 = st.slider("V₂ (L)", 1.0, 10.0, st.session_state.V2, 0.1)

    T1 = calculate_temperature(st.session_state.P1, st.session_state.V1)
    T2 = calculate_temperature(st.session_state.P2, st.session_state.V2)

    st.success(f"**T₁ = {T1:.1f} K** → **T₂ = {T2:.1f} K**")

    st.divider()

    # 경로 추가
    st.subheader("🛤️ 경로 추가")

    path_type = st.selectbox("경로 타입", ["등온", "등압", "등적", "단열"])

    if st.button("➕ 경로 추가", use_container_width=True):
        P1, V1 = st.session_state.P1, st.session_state.V1
        P2, V2 = st.session_state.P2, st.session_state.V2

        try:
            if path_type == "등온":
                P_array, V_array = generate_isothermal_path(P1, V1, P2, V2)
            elif path_type == "등압":
                P_array, V_array = generate_isobaric_path(P1, V1, V2)
            elif path_type == "등적":
                P_array, V_array = generate_isochoric_path(V1, P1, P2)
            elif path_type == "단열":
                P_array, V_array = generate_adiabatic_path(P1, V1, P2, V2, gas_type=st.session_state.gas_type)

            path = calculate_path_properties(P_array, V_array, path_type, st.session_state.gas_type)
            st.session_state.paths.append(path)
            st.success(f"✅ {path_type} 경로 추가됨!")
            st.rerun()
        except Exception as e:
            st.error(f"오류: {e}")

    st.divider()

    # 도구
    st.subheader("🔧 도구")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 경로 삭제", use_container_width=True):
            st.session_state.paths = []
            st.session_state.optimal_path = None
            st.session_state.cycle_data = None
            st.rerun()
    with col2:
        if st.button("🔄 초기화", use_container_width=True):
            st.session_state.paths = []
            st.session_state.optimal_path = None
            st.session_state.cycle_data = None
            st.session_state.P1 = 5.0
            st.session_state.V1 = 2.0
            st.session_state.P2 = 1.0
            st.session_state.V2 = 8.0
            st.rerun()

# 메인 탭
tab1, tab2, tab3, tab4 = st.tabs(["📈 경로 분석", "🔄 열역학 사이클", "🔬 3D 시각화", "📊 알고리즘 비교"])

# 탭 1: 경로 분석
with tab1:
    # 상단 설명 카드
    st.markdown("""
    <div class="glow-card fade-in">
        <h4 style="color: #00d4ff; margin-bottom: 0.8rem;">📈 경로 분석이란?</h4>
        <p style="color: #a0aec0; margin-bottom: 1rem; line-height: 1.6;">
            기체가 <strong style="color: #f472b6;">상태 A</strong>에서 <strong style="color: #4ade80;">상태 B</strong>로
            변할 때, 어떤 경로를 따라가느냐에 따라 <strong style="color: #00d4ff;">일(W)</strong>과
            <strong style="color: #7c3aed;">열(Q)</strong>이 달라집니다.
            이 시뮬레이터는 다양한 경로를 비교하고 최적의 경로를 찾아줍니다!
        </p>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <span style="color: #94a3b8;">🟢 <strong>등온</strong>: 온도 일정</span>
            <span style="color: #94a3b8;">🔵 <strong>등압</strong>: 압력 일정</span>
            <span style="color: #94a3b8;">🟣 <strong>등적</strong>: 부피 일정</span>
            <span style="color: #94a3b8;">🟠 <strong>단열</strong>: 열 출입 없음</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_main1, col_main2 = st.columns([2, 1])

    with col_main1:
        # 최적 경로 찾기 버튼
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            algorithm = st.selectbox("알고리즘", ["dijkstra", "astar"], format_func=lambda x: "Dijkstra" if x == "dijkstra" else "A*",
                                    help="Dijkstra: 정확한 최단 경로 탐색\nA*: 휴리스틱을 사용한 빠른 탐색")

        with col_btn2:
            optimization = st.selectbox("최적화 목표", ["max_work", "min_entropy", "max_efficiency"],
                                        format_func=lambda x: {"max_work": "최대 일", "min_entropy": "최소 엔트로피", "max_efficiency": "최대 효율"}[x],
                                        help="최대 일: 가장 많은 일을 하는 경로\n최소 엔트로피: 엔트로피 증가가 가장 적은 경로\n최대 효율: 효율이 가장 높은 경로")

        with col_btn3:
            if st.button("🔍 최적 경로 찾기", use_container_width=True, type="primary"):
                with st.spinner("최적 경로 탐색 중..."):
                    try:
                        optimal = find_optimal_path(
                            st.session_state.P1, st.session_state.V1,
                            st.session_state.P2, st.session_state.V2,
                            grid_size=50,
                            algorithm=algorithm,
                            optimization_target=optimization,
                            gas_type=st.session_state.gas_type
                        )
                        if optimal:
                            st.session_state.optimal_path = optimal
                            st.success(f"✅ 최적 경로 발견! W = {optimal['W']:.2f} L·atm")
                        else:
                            st.error("최적 경로를 찾을 수 없습니다.")
                    except Exception as e:
                        st.error(f"오류: {e}")

        # P-V 다이어그램
        st.markdown("""
        <div style="background: rgba(0, 212, 255, 0.05); border-left: 3px solid #00d4ff; padding: 1rem; border-radius: 0 12px 12px 0; margin-bottom: 1rem;">
            <h4 style="color: #00d4ff; margin: 0 0 0.5rem 0;">📈 P-V 다이어그램</h4>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                <strong>X축</strong>: 부피 V (L) | <strong>Y축</strong>: 압력 P (atm)<br>
                <span style="color: #4ade80;">●</span> 초기 상태 A → <span style="color: #f472b6;">■</span> 최종 상태 B<br>
                <em>곡선 아래 면적 = 기체가 한 일 (W)</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig_pv, _ = plot_pv_diagram(
            st.session_state.paths,
            st.session_state.optimal_path,
            st.session_state.P1, st.session_state.V1,
            st.session_state.P2, st.session_state.V2,
            dark_mode=st.session_state.dark_mode
        )
        st.pyplot(fig_pv)

    with col_main2:
        # 결과 요약
        st.subheader("📋 결과 요약")

        if st.session_state.optimal_path:
            opt = st.session_state.optimal_path
            st.metric("⭐ 최적 경로 일", f"{opt['W']:.2f} L·atm")
            st.metric("효율", f"{opt['efficiency']:.1f}%")
            st.metric("엔트로피 변화", f"{opt['dS']:.4f} L·atm/K")

            # 추가 정보
            with st.expander("상세 정보"):
                st.write(f"**열 (Q):** {opt['Q']:.2f} L·atm")
                st.write(f"**ΔU:** {opt['dU']:.2f} L·atm")
                st.write(f"**ΔH:** {opt.get('dH', 0):.2f} L·atm")
                st.write(f"**ΔG:** {opt.get('dG', 0):.2f} L·atm")
                st.write(f"**알고리즘:** {opt.get('algorithm', 'dijkstra').upper()}")

        # 경로 목록
        if st.session_state.paths:
            st.subheader("경로 목록")
            for i, path in enumerate(st.session_state.paths):
                with st.expander(f"경로 {i+1}: {path.get('type', '일반')}"):
                    st.write(f"**W:** {path['W']:.2f} L·atm")
                    st.write(f"**Q:** {path['Q']:.2f} L·atm")
                    st.write(f"**효율:** {path['efficiency']:.1f}%")

    # 비교 그래프 - 경로가 있을 때만 표시
    if st.session_state.paths or st.session_state.optimal_path:
        st.divider()

        # 비교 그래프 설명
        st.markdown("""
        <div class="glow-card">
            <h4 style="color: #7c3aed; margin-bottom: 0.8rem;">📊 경로별 비교 분석</h4>
            <p style="color: #a0aec0; line-height: 1.6;">
                각 경로가 얼마나 효율적인지 비교해보세요!
                <strong style="color: #00d4ff;">막대가 높을수록</strong> 해당 값이 큽니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("""
            <div style="background: rgba(0, 212, 255, 0.05); border-left: 3px solid #00d4ff; padding: 0.8rem; border-radius: 0 12px 12px 0; margin-bottom: 0.5rem;">
                <h5 style="color: #00d4ff; margin: 0 0 0.3rem 0;">⚡ 일(W) 비교</h5>
                <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">
                    기체가 팽창하면서 외부에 한 <strong>일의 양</strong><br>
                    <span style="color: #4ade80;">점선</span> = 등온 가역과정 (이론적 최대)
                </p>
            </div>
            """, unsafe_allow_html=True)
            W_rev = n * R * T1 * np.log(st.session_state.V2 / st.session_state.V1) if st.session_state.V1 > 0 and st.session_state.V2 > 0 else None
            fig_w, _ = plot_work_comparison(st.session_state.paths, st.session_state.optimal_path, W_rev, dark_mode=st.session_state.dark_mode)
            st.pyplot(fig_w)

        with col_g2:
            st.markdown("""
            <div style="background: rgba(124, 58, 237, 0.05); border-left: 3px solid #7c3aed; padding: 0.8rem; border-radius: 0 12px 12px 0; margin-bottom: 0.5rem;">
                <h5 style="color: #7c3aed; margin: 0 0 0.3rem 0;">📈 효율 비교</h5>
                <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">
                    흡수한 열 대비 한 일의 비율 <strong>(W/Q × 100%)</strong><br>
                    <span style="color: #4ade80;">100%</span> = 가역과정 (이론적 한계)
                </p>
            </div>
            """, unsafe_allow_html=True)
            fig_e, _ = plot_efficiency_comparison(st.session_state.paths, st.session_state.optimal_path, dark_mode=st.session_state.dark_mode)
            st.pyplot(fig_e)

        # 종합 비교 (Plotly)
        st.markdown("""
        <div style="background: rgba(244, 114, 182, 0.05); border-left: 3px solid #f472b6; padding: 1rem; border-radius: 0 12px 12px 0; margin: 1rem 0;">
            <h4 style="color: #f472b6; margin: 0 0 0.5rem 0;">📊 열역학적 성질 종합 비교</h4>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                <strong>W (일)</strong>: 기체가 외부에 한 일 |
                <strong>Q (열)</strong>: 기체가 흡수한 열 |
                <strong>ΔS</strong>: 엔트로피 변화 (무질서도) |
                <strong>효율</strong>: 열→일 변환율
            </p>
        </div>
        """, unsafe_allow_html=True)
        fig_props = plot_thermodynamic_properties(st.session_state.paths, st.session_state.optimal_path, st.session_state.dark_mode)
        st.plotly_chart(fig_props, use_container_width=True)

        # 결과 내보내기
        st.divider()
        csv_data = create_export_data(st.session_state.paths, st.session_state.optimal_path)
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv_data,
            file_name="thermodynamics_results.csv",
            mime="text/csv"
        )

# 탭 2: 열역학 사이클
with tab2:
    # 사이클 설명 카드
    st.markdown("""
    <div class="glow-card fade-in">
        <h4 style="color: #f472b6; margin-bottom: 0.8rem;">🔄 열역학 사이클이란?</h4>
        <p style="color: #a0aec0; margin-bottom: 1rem; line-height: 1.6;">
            열역학 사이클은 <strong style="color: #00d4ff;">열에너지를 일로 변환</strong>하는 과정입니다.
            자동차 엔진, 발전소, 냉장고 등이 모두 열역학 사이클을 이용합니다!
        </p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px;">
                <strong style="color: #fb923c;">🚗 오토 사이클</strong>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 5px 0 0 0;">가솔린 엔진의 원리</p>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px;">
                <strong style="color: #22d3d1;">🚛 디젤 사이클</strong>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 5px 0 0 0;">디젤 엔진의 원리</p>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px;">
                <strong style="color: #a78bfa;">✈️ 브레이턴 사이클</strong>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 5px 0 0 0;">제트엔진, 가스터빈</p>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px;">
                <strong style="color: #4ade80;">⚡ 카르노 사이클</strong>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 5px 0 0 0;">이론적 최대 효율</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cycle1, col_cycle2 = st.columns([1, 2])

    with col_cycle1:
        cycle_type = st.selectbox("사이클 선택", ["Otto (오토)", "Diesel (디젤)", "Brayton (브레이턴)", "Carnot (카르노)"])

        if "Otto" in cycle_type:
            st.markdown("""
            <div style="background: rgba(251, 146, 60, 0.1); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                <strong style="color: #fb923c;">🚗 오토 사이클</strong>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                    가솔린 엔진의 작동 원리!<br>
                    압축비가 높을수록 효율이 좋아요.
                </p>
            </div>
            """, unsafe_allow_html=True)
            compression_ratio = st.slider("압축비 (r)", 5.0, 15.0, 8.0, 0.5)
            heat_added = st.slider("추가 열량 (L·atm)", 10.0, 100.0, 50.0, 5.0)

            if st.button("🔄 오토 사이클 생성", use_container_width=True):
                try:
                    cycle = generate_otto_cycle(
                        V1=st.session_state.V1 * 2,
                        V2=st.session_state.V1,
                        P1=1.0,
                        compression_ratio=compression_ratio,
                        heat_added=heat_added,
                        gas_type=st.session_state.gas_type
                    )
                    st.session_state.cycle_data = cycle
                    st.success(f"✅ 오토 사이클 생성! 효율: {cycle['efficiency']:.1f}%")
                except Exception as e:
                    st.error(f"오류: {e}")

        elif "Diesel" in cycle_type:
            st.markdown("**디젤 사이클 (디젤 엔진)**")
            compression_ratio = st.slider("압축비 (r)", 10.0, 25.0, 18.0, 0.5)
            cutoff_ratio = st.slider("차단비 (rc)", 1.5, 4.0, 2.5, 0.1)

            if st.button("🔄 디젤 사이클 생성", use_container_width=True):
                try:
                    cycle = generate_diesel_cycle(
                        V1=st.session_state.V1 * 2,
                        P1=1.0,
                        compression_ratio=compression_ratio,
                        cutoff_ratio=cutoff_ratio,
                        gas_type=st.session_state.gas_type
                    )
                    st.session_state.cycle_data = cycle
                    st.success(f"✅ 디젤 사이클 생성! 효율: {cycle['efficiency']:.1f}%")
                except Exception as e:
                    st.error(f"오류: {e}")

        elif "Brayton" in cycle_type:
            st.markdown("**브레이턴 사이클 (가스 터빈)**")
            pressure_ratio = st.slider("압력비 (rp)", 5.0, 20.0, 10.0, 0.5)
            T_max = st.slider("최고 온도 (K)", 800.0, 1500.0, 1200.0, 50.0)

            if st.button("🔄 브레이턴 사이클 생성", use_container_width=True):
                try:
                    cycle = generate_brayton_cycle(
                        P1=1.0,
                        T1=300.0,
                        pressure_ratio=pressure_ratio,
                        T3=T_max,
                        gas_type='diatomic'
                    )
                    st.session_state.cycle_data = cycle
                    st.success(f"✅ 브레이턴 사이클 생성! 효율: {cycle['efficiency']:.1f}%")
                except Exception as e:
                    st.error(f"오류: {e}")

        elif "Carnot" in cycle_type:
            st.markdown("**카르노 사이클 (이론적 최대 효율)**")
            T_hot = st.slider("고온부 온도 (K)", 400.0, 1000.0, 600.0, 10.0)
            T_cold = st.slider("저온부 온도 (K)", 200.0, 400.0, 300.0, 10.0)

            if st.button("🔄 카르노 사이클 생성", use_container_width=True):
                try:
                    cycle = generate_carnot_cycle(
                        P1=st.session_state.P1,
                        V1=st.session_state.V1,
                        T_hot=T_hot,
                        T_cold=T_cold,
                        gas_type=st.session_state.gas_type
                    )
                    st.session_state.cycle_data = cycle
                    st.success(f"✅ 카르노 사이클 생성! 효율: {cycle['efficiency']:.1f}%")
                except Exception as e:
                    st.error(f"오류: {e}")

    with col_cycle2:
        if st.session_state.cycle_data:
            cycle = st.session_state.cycle_data

            # 사이클 다이어그램
            fig_cycle = plot_cycle_diagram(cycle, st.session_state.dark_mode)
            st.plotly_chart(fig_cycle, use_container_width=True)

            # 사이클 정보
            st.markdown("### 📊 사이클 성능")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("효율", f"{cycle['efficiency']:.1f}%")
            with col_m2:
                st.metric("순일", f"{cycle['W_net']:.2f} L·atm")
            with col_m3:
                st.metric("흡수 열", f"{cycle['Q_in']:.2f} L·atm")
            with col_m4:
                st.metric("방출 열", f"{cycle['Q_out']:.2f} L·atm")

            # 상태점 테이블
            st.markdown("### 📋 상태점")
            states = cycle['states']
            state_df = pd.DataFrame([
                {"상태": name, "P (atm)": f"{s['P']:.2f}", "V (L)": f"{s['V']:.2f}", "T (K)": f"{s['T']:.1f}"}
                for name, s in states.items()
            ])
            st.dataframe(state_df, use_container_width=True, hide_index=True)
        else:
            st.info("👈 왼쪽에서 사이클을 선택하고 생성해주세요.")

# 탭 3: 3D 시각화
with tab3:
    # 3D 설명 카드
    st.markdown("""
    <div class="glow-card fade-in">
        <h4 style="color: #22d3d1; margin-bottom: 0.8rem;">🔬 3D P-V-T 다이어그램이란?</h4>
        <p style="color: #a0aec0; margin-bottom: 1rem; line-height: 1.6;">
            이상기체 상태방정식 <strong style="color: #00d4ff;">PV = nRT</strong>를 3차원으로 시각화합니다!
            <strong style="color: #f472b6;">압력(P)</strong>, <strong style="color: #4ade80;">부피(V)</strong>,
            <strong style="color: #fb923c;">온도(T)</strong> 세 변수의 관계를 한눈에 볼 수 있어요.
        </p>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            💡 <strong>반투명 표면</strong>은 이상기체가 존재할 수 있는 모든 상태를 나타냅니다.
            경로는 이 표면 위를 따라 이동해요!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_3d1, col_3d2 = st.columns([3, 1])

    with col_3d1:
        show_surface = st.checkbox("상태방정식 표면 표시", value=True, help="PV=nRT 표면을 보여줍니다")

        fig_3d = plot_3d_pvt_diagram(
            st.session_state.paths,
            st.session_state.optimal_path,
            st.session_state.P1, st.session_state.V1,
            st.session_state.P2, st.session_state.V2,
            show_surface=show_surface,
            dark_mode=st.session_state.dark_mode
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_3d2:
        st.markdown("""
        <div style="background: rgba(34, 211, 209, 0.1); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <h5 style="color: #22d3d1; margin: 0 0 0.5rem 0;">🎮 마우스 조작법</h5>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.8;">
                🖱️ <strong>드래그</strong>: 회전<br>
                🔍 <strong>스크롤</strong>: 확대/축소<br>
                ⌨️ <strong>Shift+드래그</strong>: 이동
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(124, 58, 237, 0.1); border-radius: 12px; padding: 1rem;">
            <h5 style="color: #7c3aed; margin: 0 0 0.5rem 0;">📌 범례</h5>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.8;">
                <span style="color: #4ade80;">●</span> 초기 상태 A<br>
                <span style="color: #f472b6;">■</span> 최종 상태 B<br>
                🌈 반투명 표면 = PV=nRT
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.paths or st.session_state.optimal_path:
            st.markdown("### 📊 현재 경로")
            for i, path in enumerate(st.session_state.paths):
                st.write(f"경로 {i+1}: {path.get('type', '일반')}")
            if st.session_state.optimal_path:
                st.write("⭐ 최적 경로")

# 탭 4: 알고리즘 비교
with tab4:
    # 알고리즘 설명 카드
    st.markdown("""
    <div class="glow-card fade-in">
        <h4 style="color: #fb923c; margin-bottom: 0.8rem;">📊 경로 탐색 알고리즘 비교</h4>
        <p style="color: #a0aec0; margin-bottom: 1rem; line-height: 1.6;">
            최적 경로를 찾는 두 가지 알고리즘의 성능을 비교해보세요!
            같은 결과를 찾지만 <strong style="color: #00d4ff;">속도</strong>가 다를 수 있어요.
        </p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 12px; border-left: 3px solid #00d4ff;">
                <strong style="color: #00d4ff;">🔵 Dijkstra 알고리즘</strong>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 8px 0 0 0; line-height: 1.5;">
                    모든 경로를 탐색하여<br>
                    <strong>정확한 최적해</strong>를 보장<br>
                    느리지만 확실함!
                </p>
            </div>
            <div style="background: rgba(124, 58, 237, 0.1); padding: 15px; border-radius: 12px; border-left: 3px solid #7c3aed;">
                <strong style="color: #7c3aed;">🟣 A* 알고리즘</strong>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 8px 0 0 0; line-height: 1.5;">
                    휴리스틱(예측)을 사용해<br>
                    <strong>빠르게 탐색</strong><br>
                    똑똑하고 효율적!
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    grid_size = st.slider("격자 크기", 20, 100, 50, 10)

    if st.button("🔬 알고리즘 비교 실행", use_container_width=True, type="primary"):
        with st.spinner("알고리즘 비교 중..."):
            try:
                results = compare_algorithms(
                    st.session_state.P1, st.session_state.V1,
                    st.session_state.P2, st.session_state.V2,
                    grid_size=grid_size,
                    gas_type=st.session_state.gas_type
                )

                # 결과 시각화
                fig_compare = plot_algorithm_comparison(results, st.session_state.dark_mode)
                st.plotly_chart(fig_compare, use_container_width=True)

                # 상세 결과 테이블
                st.markdown("### 📋 상세 결과")
                result_data = []
                for algo_name, data in results.items():
                    result_data.append({
                        "알고리즘": algo_name.replace("_", " ").title(),
                        "계산 시간 (초)": f"{data['time']:.4f}",
                        "찾은 일 (L·atm)": f"{data['result']['W']:.2f}" if data['result'] else "N/A",
                        "효율 (%)": f"{data['result']['efficiency']:.1f}" if data['result'] else "N/A"
                    })
                st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)

                # 결론
                best_algo = min(results.items(), key=lambda x: x[1]['time'])
                best_work = max(results.items(), key=lambda x: x[1]['result']['W'] if x[1]['result'] else 0)

                st.success(f"""
                **결론:**
                - 가장 빠른 알고리즘: **{best_algo[0].replace('_', ' ').title()}** ({best_algo[1]['time']:.4f}초)
                - 최대 일 발견: **{best_work[0].replace('_', ' ').title()}** ({best_work[1]['result']['W']:.2f} L·atm)
                """)

            except Exception as e:
                st.error(f"오류: {e}")

# 하단 정보
st.divider()

with st.expander("📖 사용 방법"):
    st.markdown("""
    ### 기본 사용법
    1. **기체 타입 선택**: 사이드바에서 단원자/이원자/다원자 기체를 선택합니다.
    2. **상태 설정**: 초기 상태 A와 최종 상태 B의 압력과 부피를 설정합니다.
    3. **경로 추가**: 등온/등압/등적/단열 경로를 추가합니다.
    4. **최적 경로 찾기**: Dijkstra 또는 A* 알고리즘으로 최적 경로를 찾습니다.

    ### 탭 설명
    - **경로 분석**: P-V 다이어그램과 경로별 비교
    - **열역학 사이클**: 오토/디젤/브레이턴/카르노 사이클 시뮬레이션
    - **3D 시각화**: P-V-T 공간에서 경로 시각화
    - **알고리즘 비교**: Dijkstra vs A* 성능 비교
    """)

with st.expander("🔬 열역학 상수"):
    st.markdown(f"""
    | 상수 | 값 | 설명 |
    |------|-----|------|
    | R | {R} L·atm/(mol·K) | 이상기체 상수 |
    | n | {n} mol | 몰수 |
    | γ (단원자) | 5/3 ≈ 1.667 | 비열비 |
    | γ (이원자) | 7/5 = 1.4 | 비열비 |
    | γ (다원자) | 4/3 ≈ 1.333 | 비열비 |
    """)

st.markdown("""
<div class="footer">
    <p class="footer-title">
        열역학 경로 최적화 시뮬레이터 v3.0
    </p>
    <p class="footer-subtitle">
        2025 고급화학 주제발표 프로젝트
    </p>
    <p class="footer-tech">
        Built with Streamlit · NumPy · Matplotlib · Plotly
    </p>
    <div style="margin-top: 1.5rem;">
        <span class="status-badge active">Live</span>
        <span class="status-badge active">Interactive</span>
        <span class="status-badge active">Scientific</span>
    </div>
</div>
""", unsafe_allow_html=True)
