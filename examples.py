"""
예제 시나리오 모듈 (Enhanced Version)
발표용 미리 정의된 예제들
"""

import numpy as np
from thermodynamics import (
    generate_isothermal_path,
    generate_isobaric_path,
    generate_isochoric_path,
    generate_adiabatic_path,
    calculate_path_properties,
    generate_otto_cycle,
    generate_diesel_cycle,
    generate_brayton_cycle,
    generate_carnot_cycle
)


def example_1_isothermal_vs_adiabatic(gas_type='monatomic'):
    """
    예제 1: 등온 vs 단열 비교
    A(5 atm, 2L) → B(1 atm, 10L)
    """
    P1, V1 = 5, 2
    P2, V2 = 1, 10

    # 등온 경로
    P_iso, V_iso = generate_isothermal_path(P1, V1, P2, V2)
    path_iso = calculate_path_properties(P_iso, V_iso, "등온", gas_type)

    # 단열 경로
    P_adi, V_adi = generate_adiabatic_path(P1, V1, P2, V2, gas_type=gas_type)
    path_adi = calculate_path_properties(P_adi, V_adi, "단열", gas_type)

    return {
        'P1': P1, 'V1': V1, 'P2': P2, 'V2': V2,
        'paths': [path_iso, path_adi],
        'description': '등온 과정과 단열 과정의 일 비교'
    }


def example_2_carnot_cycle(gas_type='monatomic'):
    """
    예제 2: 카르노 사이클 (4단계)
    등온 팽창 → 단열 팽창 → 등온 압축 → 단열 압축
    """
    P_A, V_A = 5, 2
    P_B, V_B = 2, 5
    P_C, V_C = 1, 8
    P_D, V_D = 2, 4

    paths = []

    # 등온 팽창 (A → B)
    P1, V1 = generate_isothermal_path(P_A, V_A, P_B, V_B)
    path1 = calculate_path_properties(P1, V1, "등온", gas_type)
    path1['type'] = "등온 팽창"
    paths.append(path1)

    # 단열 팽창 (B → C)
    P2, V2 = generate_adiabatic_path(P_B, V_B, P_C, V_C, gas_type=gas_type)
    path2 = calculate_path_properties(P2, V2, "단열", gas_type)
    path2['type'] = "단열 팽창"
    paths.append(path2)

    # 등온 압축 (C → D)
    P3, V3 = generate_isothermal_path(P_C, V_C, P_D, V_D)
    path3 = calculate_path_properties(P3, V3, "등온", gas_type)
    path3['type'] = "등온 압축"
    paths.append(path3)

    # 단열 압축 (D → A)
    P4, V4 = generate_adiabatic_path(P_D, V_D, P_A, V_A, gas_type=gas_type)
    path4 = calculate_path_properties(P4, V4, "단열", gas_type)
    path4['type'] = "단열 압축"
    paths.append(path4)

    return {
        'P1': P_A, 'V1': V_A, 'P2': P_A, 'V2': V_A,
        'paths': paths,
        'is_cycle': True,
        'description': '카르노 사이클 - 이론적 최대 효율 열기관'
    }


def example_3_inefficient_path(gas_type='monatomic'):
    """
    예제 3: 비효율적 경로 vs 최적 경로
    지그재그 경로를 만들어서 최적 경로와 비교
    """
    P1, V1 = 5, 2
    P2, V2 = 1, 8

    # 비효율적 경로: 등적 → 등압 (두 단계)
    # 1단계: 등적 (V=2에서 P=5→1)
    P_step1, V_step1 = generate_isochoric_path(V1, P1, P2)
    # 2단계: 등압 (P=1에서 V=2→8)
    P_step2, V_step2 = generate_isobaric_path(P2, V1, V2)

    # 경로 합치기
    P_combined = np.concatenate([P_step1, P_step2[1:]])
    V_combined = np.concatenate([V_step1, V_step2[1:]])
    path_ineff = calculate_path_properties(P_combined, V_combined, "일반", gas_type)
    path_ineff['type'] = "등적+등압 (비효율)"

    # 등온 경로 (효율적)
    P_iso, V_iso = generate_isothermal_path(P1, V1, P2, V2)
    path_iso = calculate_path_properties(P_iso, V_iso, "등온", gas_type)
    path_iso['type'] = "등온 (효율적)"

    return {
        'P1': P1, 'V1': V1, 'P2': P2, 'V2': V2,
        'paths': [path_ineff, path_iso],
        'description': '비효율적 경로와 효율적 경로 비교'
    }


def example_4_multiple_paths(gas_type='monatomic'):
    """
    예제 4: 여러 경로 동시 비교
    등온, 등압, 등적+등압, 단열 모두 비교
    """
    P1, V1 = 5, 2
    P2, V2 = 1, 8

    paths = []

    # 등온
    P_iso, V_iso = generate_isothermal_path(P1, V1, P2, V2)
    paths.append(calculate_path_properties(P_iso, V_iso, "등온", gas_type))

    # 등압 (P1에서 V1→V2, 하지만 P2로 안 감)
    P_isoP, V_isoP = generate_isobaric_path(P1, V1, V2)
    path_isoP = calculate_path_properties(P_isoP, V_isoP, "등압", gas_type)
    path_isoP['type'] = "등압 (P=5atm)"
    paths.append(path_isoP)

    # 등적+등압 (중간점 경유)
    P_mid = (P1 + P2) / 2
    P_isoV1, V_isoV1 = generate_isochoric_path(V1, P1, P_mid)
    P_isoP_mid, V_isoP_mid = generate_isobaric_path(P_mid, V1, V2)
    P_combined = np.concatenate([P_isoV1, P_isoP_mid[1:]])
    V_combined = np.concatenate([V_isoV1, V_isoP_mid[1:]])
    path_combined = calculate_path_properties(P_combined, V_combined, "일반", gas_type)
    path_combined['type'] = "등적+등압"
    paths.append(path_combined)

    # 단열
    P_adi, V_adi = generate_adiabatic_path(P1, V1, P2, V2, gas_type=gas_type)
    paths.append(calculate_path_properties(P_adi, V_adi, "단열", gas_type))

    return {
        'P1': P1, 'V1': V1, 'P2': P2, 'V2': V2,
        'paths': paths,
        'description': '다양한 열역학 경로 종합 비교'
    }


def example_5_otto_cycle(compression_ratio=8.0, heat_added=50.0, gas_type='diatomic'):
    """
    예제 5: 오토 사이클 (가솔린 엔진)
    """
    return generate_otto_cycle(
        V1=8.0,
        V2=1.0,
        P1=1.0,
        compression_ratio=compression_ratio,
        heat_added=heat_added,
        gas_type=gas_type
    )


def example_6_diesel_cycle(compression_ratio=18.0, cutoff_ratio=2.5, gas_type='diatomic'):
    """
    예제 6: 디젤 사이클 (디젤 엔진)
    """
    return generate_diesel_cycle(
        V1=10.0,
        P1=1.0,
        compression_ratio=compression_ratio,
        cutoff_ratio=cutoff_ratio,
        gas_type=gas_type
    )


def example_7_brayton_cycle(pressure_ratio=10.0, T_max=1200.0, gas_type='diatomic'):
    """
    예제 7: 브레이턴 사이클 (가스 터빈)
    """
    return generate_brayton_cycle(
        P1=1.0,
        T1=300.0,
        pressure_ratio=pressure_ratio,
        T3=T_max,
        gas_type=gas_type
    )


def example_8_carnot_ideal(T_hot=600.0, T_cold=300.0, gas_type='monatomic'):
    """
    예제 8: 이상적인 카르노 사이클
    """
    return generate_carnot_cycle(
        P1=5.0,
        V1=2.0,
        T_hot=T_hot,
        T_cold=T_cold,
        gas_type=gas_type
    )


def example_9_gas_type_comparison():
    """
    예제 9: 기체 타입별 비교
    동일한 경로에서 단원자/이원자/다원자 기체의 차이
    """
    P1, V1 = 5, 2
    P2, V2 = 1, 8

    results = {}

    for gas_type in ['monatomic', 'diatomic', 'polyatomic']:
        P_adi, V_adi = generate_adiabatic_path(P1, V1, P2, V2, gas_type=gas_type)
        path = calculate_path_properties(P_adi, V_adi, "단열", gas_type)
        results[gas_type] = path

    return {
        'P1': P1, 'V1': V1, 'P2': P2, 'V2': V2,
        'results': results,
        'description': '기체 타입에 따른 단열 과정 비교'
    }


def example_10_efficiency_comparison():
    """
    예제 10: 사이클 효율 비교
    동일 조건에서 오토, 디젤, 브레이턴 사이클 효율 비교
    """
    # 동일한 압축비 10으로 비교
    compression_ratio = 10.0

    results = {
        'otto': generate_otto_cycle(
            V1=10.0, V2=1.0, P1=1.0,
            compression_ratio=compression_ratio,
            heat_added=50.0,
            gas_type='diatomic'
        ),
        'diesel': generate_diesel_cycle(
            V1=10.0, P1=1.0,
            compression_ratio=compression_ratio,
            cutoff_ratio=2.0,
            gas_type='diatomic'
        ),
        'brayton': generate_brayton_cycle(
            P1=1.0, T1=300.0,
            pressure_ratio=compression_ratio,
            T3=1200.0,
            gas_type='diatomic'
        )
    }

    return {
        'results': results,
        'compression_ratio': compression_ratio,
        'description': '열기관 사이클 효율 비교 (동일 압축비)'
    }
