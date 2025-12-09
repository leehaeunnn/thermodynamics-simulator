"""
열역학 경로 최적화 시뮬레이터 (Enhanced Version)
Streamlit 메인 앱

새로운 기능:
- 3D P-V-T 다이어그램
- 다양한 기체 타입 지원
- 열역학 사이클 (오토, 디젤, 브레이턴, 카르노)
- A* 알고리즘 경로 탐색
- 결과 내보내기
- 다크모드
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

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
    }
</style>
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

# 제목
st.markdown('<h1 class="main-header">⚛️ 열역학 경로 최적화 시뮬레이터</h1>', unsafe_allow_html=True)

# 설명
st.markdown("""
**이상기체가 A 상태에서 B 상태로 변할 때, 다양한 경로를 비교하고 최적 경로를 찾습니다.**

| 물리량 | 설명 |
|--------|------|
| **W** | 시스템이 한 일 |
| **Q** | 시스템이 흡수한 열 |
| **ΔU** | 내부에너지 변화 |
| **ΔH** | 엔탈피 변화 |
| **ΔS** | 엔트로피 변화 |
| **ΔG** | 깁스 자유에너지 변화 |
""")

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
    col_main1, col_main2 = st.columns([2, 1])

    with col_main1:
        # 최적 경로 찾기 버튼
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            algorithm = st.selectbox("알고리즘", ["dijkstra", "astar"], format_func=lambda x: "Dijkstra" if x == "dijkstra" else "A*")

        with col_btn2:
            optimization = st.selectbox("최적화 목표", ["max_work", "min_entropy", "max_efficiency"],
                                        format_func=lambda x: {"max_work": "최대 일", "min_entropy": "최소 엔트로피", "max_efficiency": "최대 효율"}[x])

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
        st.subheader("📈 P-V 다이어그램")
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

    # 비교 그래프
    if st.session_state.paths or st.session_state.optimal_path:
        st.divider()
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("일 비교")
            W_rev = n * R * T1 * np.log(st.session_state.V2 / st.session_state.V1) if st.session_state.V1 > 0 and st.session_state.V2 > 0 else None
            fig_w, _ = plot_work_comparison(st.session_state.paths, st.session_state.optimal_path, W_rev, dark_mode=st.session_state.dark_mode)
            st.pyplot(fig_w)

        with col_g2:
            st.subheader("효율 비교")
            fig_e, _ = plot_efficiency_comparison(st.session_state.paths, st.session_state.optimal_path, dark_mode=st.session_state.dark_mode)
            st.pyplot(fig_e)

        # 종합 비교 (Plotly)
        st.subheader("📊 열역학적 성질 종합 비교")
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
    st.subheader("🔄 열역학 사이클 시뮬레이션")

    col_cycle1, col_cycle2 = st.columns([1, 2])

    with col_cycle1:
        cycle_type = st.selectbox("사이클 선택", ["Otto (오토)", "Diesel (디젤)", "Brayton (브레이턴)", "Carnot (카르노)"])

        if "Otto" in cycle_type:
            st.markdown("**오토 사이클 (가솔린 엔진)**")
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
    st.subheader("🔬 3D P-V-T 다이어그램")

    col_3d1, col_3d2 = st.columns([3, 1])

    with col_3d1:
        show_surface = st.checkbox("상태방정식 표면 표시", value=True)

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
        st.markdown("### 🎮 조작법")
        st.markdown("""
        - **회전**: 드래그
        - **줌**: 스크롤
        - **이동**: Shift + 드래그

        ### 📌 범례
        - 🟢 초기 상태 A
        - 🔴 최종 상태 B
        - 반투명 표면: PV=nRT
        """)

        if st.session_state.paths or st.session_state.optimal_path:
            st.markdown("### 📊 현재 경로")
            for i, path in enumerate(st.session_state.paths):
                st.write(f"경로 {i+1}: {path.get('type', '일반')}")
            if st.session_state.optimal_path:
                st.write("⭐ 최적 경로")

# 탭 4: 알고리즘 비교
with tab4:
    st.subheader("📊 경로 탐색 알고리즘 비교")

    st.markdown("""
    Dijkstra와 A* 알고리즘의 성능을 비교합니다.
    A* 알고리즘은 휴리스틱 함수에 따라 다른 성능을 보입니다.
    """)

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

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    열역학 경로 최적화 시뮬레이터 v2.0 | Made with Streamlit
</div>
""", unsafe_allow_html=True)
