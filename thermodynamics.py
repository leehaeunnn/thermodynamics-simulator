"""
열역학 계산 모듈 (Enhanced Version)
이상기체의 다양한 열역학 과정에 대한 계산 함수들
- 다양한 기체 타입 지원 (단원자, 이원자, 다원자)
- 열역학 사이클 (오토, 디젤, 브레이턴)
- 엔탈피, 깁스 자유에너지 계산
"""

import numpy as np
from scipy import integrate
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

# 상수 정의
R = 0.0821  # L·atm/(mol·K) - 이상기체 상수
R_J = 8.314  # J/(mol·K) - SI 단위

# 기체 타입별 비열비 및 열용량
GAS_TYPES = {
    'monatomic': {  # 단원자 (He, Ne, Ar)
        'gamma': 5/3,
        'Cv': (3/2) * R,
        'Cp': (5/2) * R,
        'name': '단원자 기체',
        'examples': 'He, Ne, Ar'
    },
    'diatomic': {  # 이원자 (N2, O2, H2)
        'gamma': 7/5,
        'Cv': (5/2) * R,
        'Cp': (7/2) * R,
        'name': '이원자 기체',
        'examples': 'N₂, O₂, H₂'
    },
    'polyatomic': {  # 다원자 (CO2, H2O)
        'gamma': 4/3,
        'Cv': 3 * R,
        'Cp': 4 * R,
        'name': '다원자 기체',
        'examples': 'CO₂, H₂O'
    }
}

# 기본값
n = 1.0  # mol - 몰수 (기본값)


def get_gas_properties(gas_type: str = 'monatomic') -> Dict:
    """기체 타입에 따른 물성치 반환"""
    return GAS_TYPES.get(gas_type, GAS_TYPES['monatomic'])


def calculate_temperature(P: float, V: float, mol: float = n) -> float:
    """
    이상기체 법칙으로 온도 계산
    PV = nRT → T = PV/(nR)
    """
    if mol == 0 or R == 0:
        return 0
    return (P * V) / (mol * R)


def calculate_pressure(T: float, V: float, mol: float = n) -> float:
    """
    이상기체 법칙으로 압력 계산
    PV = nRT → P = nRT/V
    """
    if V == 0:
        return 0
    return (mol * R * T) / V


def calculate_volume(T: float, P: float, mol: float = n) -> float:
    """
    이상기체 법칙으로 부피 계산
    PV = nRT → V = nRT/P
    """
    if P == 0:
        return 0
    return (mol * R * T) / P


# ==================== 일 계산 함수들 ====================

def calculate_work_isothermal(P1: float, V1: float, P2: float, V2: float, mol: float = n) -> float:
    """등온 과정의 일: W = nRT ln(V2/V1)"""
    T = calculate_temperature(P1, V1, mol)
    if V2 <= 0 or V1 <= 0:
        return 0
    return mol * R * T * np.log(V2 / V1)


def calculate_work_isobaric(P: float, V1: float, V2: float) -> float:
    """등압 과정의 일: W = P(V2 - V1)"""
    return P * (V2 - V1)


def calculate_work_isochoric(V: float, P1: float, P2: float) -> float:
    """등적 과정의 일: W = 0"""
    return 0.0


def calculate_work_adiabatic(P1: float, V1: float, P2: float, V2: float,
                             gas_type: str = 'monatomic', mol: float = n) -> float:
    """단열 과정의 일: W = -ΔU = -nCv(T2 - T1)"""
    gas = get_gas_properties(gas_type)
    Cv = gas['Cv']
    T1 = calculate_temperature(P1, V1, mol)
    T2 = calculate_temperature(P2, V2, mol)
    dU = mol * Cv * (T2 - T1)
    return -dU


def calculate_work_general(P_array: np.ndarray, V_array: np.ndarray) -> float:
    """일반 경로의 일 (수치적분): W = ∫P dV"""
    if len(P_array) < 2 or len(V_array) < 2:
        return 0.0
    return integrate.trapezoid(P_array, V_array)


# ==================== 열역학적 성질 변화 계산 ====================

def calculate_internal_energy_change(P1: float, V1: float, P2: float, V2: float,
                                     gas_type: str = 'monatomic', mol: float = n) -> float:
    """내부에너지 변화: ΔU = nCv(T2 - T1)"""
    gas = get_gas_properties(gas_type)
    Cv = gas['Cv']
    T1 = calculate_temperature(P1, V1, mol)
    T2 = calculate_temperature(P2, V2, mol)
    return mol * Cv * (T2 - T1)


def calculate_enthalpy_change(P1: float, V1: float, P2: float, V2: float,
                              gas_type: str = 'monatomic', mol: float = n) -> float:
    """엔탈피 변화: ΔH = nCp(T2 - T1)"""
    gas = get_gas_properties(gas_type)
    Cp = gas['Cp']
    T1 = calculate_temperature(P1, V1, mol)
    T2 = calculate_temperature(P2, V2, mol)
    return mol * Cp * (T2 - T1)


def calculate_heat(dU: float, W: float) -> float:
    """열 (제1법칙): Q = ΔU + W"""
    return dU + W


def calculate_entropy_change(P1: float, V1: float, P2: float, V2: float,
                             gas_type: str = 'monatomic', mol: float = n) -> float:
    """
    엔트로피 변화: ΔS = nR ln(V2/V1) + nCv ln(T2/T1)
    """
    gas = get_gas_properties(gas_type)
    Cv = gas['Cv']
    T1 = calculate_temperature(P1, V1, mol)
    T2 = calculate_temperature(P2, V2, mol)

    if V1 <= 0 or V2 <= 0 or T1 <= 0 or T2 <= 0:
        return 0.0

    dS_volume = mol * R * np.log(V2 / V1)
    dS_temperature = mol * Cv * np.log(T2 / T1)
    return dS_volume + dS_temperature


def calculate_gibbs_free_energy_change(P1: float, V1: float, P2: float, V2: float,
                                       gas_type: str = 'monatomic', mol: float = n) -> float:
    """
    깁스 자유에너지 변화: ΔG = ΔH - TΔS
    (등온 과정 기준으로 계산)
    """
    T1 = calculate_temperature(P1, V1, mol)
    dH = calculate_enthalpy_change(P1, V1, P2, V2, gas_type, mol)
    dS = calculate_entropy_change(P1, V1, P2, V2, gas_type, mol)
    return dH - T1 * dS


def calculate_helmholtz_free_energy_change(P1: float, V1: float, P2: float, V2: float,
                                           gas_type: str = 'monatomic', mol: float = n) -> float:
    """
    헬름홀츠 자유에너지 변화: ΔA = ΔU - TΔS
    """
    T1 = calculate_temperature(P1, V1, mol)
    dU = calculate_internal_energy_change(P1, V1, P2, V2, gas_type, mol)
    dS = calculate_entropy_change(P1, V1, P2, V2, gas_type, mol)
    return dU - T1 * dS


def calculate_efficiency(W: float, W_reversible: float) -> float:
    """효율 계산: η = (W / W_reversible) * 100"""
    if W_reversible == 0:
        return 0.0
    return (W / W_reversible) * 100


# ==================== 경로 생성 함수들 ====================

def generate_isothermal_path(P1: float, V1: float, P2: float, V2: float,
                             num_points: int = 100, mol: float = n) -> Tuple[np.ndarray, np.ndarray]:
    """등온 경로 생성: PV = constant"""
    T = calculate_temperature(P1, V1, mol)
    V_array = np.linspace(V1, V2, num_points)
    P_array = (mol * R * T) / V_array
    return P_array, V_array


def generate_isobaric_path(P: float, V1: float, V2: float,
                           num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """등압 경로 생성: P = constant"""
    V_array = np.linspace(V1, V2, num_points)
    P_array = np.full(num_points, P)
    return P_array, V_array


def generate_isochoric_path(V: float, P1: float, P2: float,
                            num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """등적 경로 생성: V = constant"""
    P_array = np.linspace(P1, P2, num_points)
    V_array = np.full(num_points, V)
    return P_array, V_array


def generate_adiabatic_path(P1: float, V1: float, P2: float, V2: float,
                            num_points: int = 100, gas_type: str = 'monatomic') -> Tuple[np.ndarray, np.ndarray]:
    """단열 경로 생성: PV^γ = constant"""
    gas = get_gas_properties(gas_type)
    gamma = gas['gamma']
    constant = P1 * (V1 ** gamma)
    V_array = np.linspace(V1, V2, num_points)
    P_array = constant / (V_array ** gamma)
    return P_array, V_array


def generate_polytropic_path(P1: float, V1: float, P2: float, V2: float,
                             n_poly: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """폴리트로픽 경로 생성: PV^n = constant"""
    constant = P1 * (V1 ** n_poly)
    V_array = np.linspace(V1, V2, num_points)
    P_array = constant / (V_array ** n_poly)
    return P_array, V_array


# ==================== 열역학 사이클 ====================

def generate_otto_cycle(V1: float, V2: float, P1: float, compression_ratio: float,
                        heat_added: float, gas_type: str = 'monatomic',
                        num_points: int = 50, mol: float = n) -> Dict:
    """
    오토 사이클 (가솔린 엔진)
    1→2: 단열 압축
    2→3: 등적 가열
    3→4: 단열 팽창
    4→1: 등적 방열

    Args:
        V1: 최대 부피 (BDC)
        V2: 최소 부피 (TDC)
        P1: 초기 압력
        compression_ratio: 압축비 (V1/V2)
        heat_added: 추가된 열
        gas_type: 기체 타입
    """
    gas = get_gas_properties(gas_type)
    gamma = gas['gamma']
    Cv = gas['Cv']

    # 상태 1: 초기 상태
    T1 = calculate_temperature(P1, V1, mol)

    # 상태 2: 단열 압축 후
    V2_calc = V1 / compression_ratio
    T2 = T1 * (compression_ratio ** (gamma - 1))
    P2 = calculate_pressure(T2, V2_calc, mol)

    # 상태 3: 등적 가열 후
    T3 = T2 + heat_added / (mol * Cv)
    P3 = calculate_pressure(T3, V2_calc, mol)

    # 상태 4: 단열 팽창 후
    T4 = T3 / (compression_ratio ** (gamma - 1))
    P4 = calculate_pressure(T4, V1, mol)

    # 경로 생성
    paths = []

    # 1→2: 단열 압축
    P_12, V_12 = generate_adiabatic_path(P1, V1, P2, V2_calc, num_points, gas_type)
    paths.append({'P': P_12, 'V': V_12, 'type': '단열 압축', 'step': '1→2'})

    # 2→3: 등적 가열
    P_23, V_23 = generate_isochoric_path(V2_calc, P2, P3, num_points)
    paths.append({'P': P_23, 'V': V_23, 'type': '등적 가열', 'step': '2→3'})

    # 3→4: 단열 팽창
    P_34, V_34 = generate_adiabatic_path(P3, V2_calc, P4, V1, num_points, gas_type)
    paths.append({'P': P_34, 'V': V_34, 'type': '단열 팽창', 'step': '3→4'})

    # 4→1: 등적 방열
    P_41, V_41 = generate_isochoric_path(V1, P4, P1, num_points)
    paths.append({'P': P_41, 'V': V_41, 'type': '등적 방열', 'step': '4→1'})

    # 효율 계산
    efficiency = 1 - (1 / (compression_ratio ** (gamma - 1)))

    # 일 계산
    Q_in = mol * Cv * (T3 - T2)
    Q_out = mol * Cv * (T4 - T1)
    W_net = Q_in - abs(Q_out)

    return {
        'paths': paths,
        'states': {
            '1': {'P': P1, 'V': V1, 'T': T1},
            '2': {'P': P2, 'V': V2_calc, 'T': T2},
            '3': {'P': P3, 'V': V2_calc, 'T': T3},
            '4': {'P': P4, 'V': V1, 'T': T4}
        },
        'efficiency': efficiency * 100,
        'W_net': W_net,
        'Q_in': Q_in,
        'Q_out': abs(Q_out),
        'compression_ratio': compression_ratio,
        'cycle_type': 'Otto'
    }


def generate_diesel_cycle(V1: float, P1: float, compression_ratio: float,
                          cutoff_ratio: float, gas_type: str = 'monatomic',
                          num_points: int = 50, mol: float = n) -> Dict:
    """
    디젤 사이클
    1→2: 단열 압축
    2→3: 등압 팽창 (연료 분사)
    3→4: 단열 팽창
    4→1: 등적 방열
    """
    gas = get_gas_properties(gas_type)
    gamma = gas['gamma']
    Cv = gas['Cv']
    Cp = gas['Cp']

    # 상태 1
    T1 = calculate_temperature(P1, V1, mol)

    # 상태 2: 단열 압축 후
    V2 = V1 / compression_ratio
    T2 = T1 * (compression_ratio ** (gamma - 1))
    P2 = calculate_pressure(T2, V2, mol)

    # 상태 3: 등압 팽창 후
    V3 = V2 * cutoff_ratio
    T3 = T2 * cutoff_ratio
    P3 = P2

    # 상태 4: 단열 팽창 후
    expansion_ratio = V1 / V3
    T4 = T3 / (expansion_ratio ** (gamma - 1))
    P4 = calculate_pressure(T4, V1, mol)

    # 경로 생성
    paths = []

    # 1→2: 단열 압축
    P_12, V_12 = generate_adiabatic_path(P1, V1, P2, V2, num_points, gas_type)
    paths.append({'P': P_12, 'V': V_12, 'type': '단열 압축', 'step': '1→2'})

    # 2→3: 등압 팽창
    P_23, V_23 = generate_isobaric_path(P2, V2, V3, num_points)
    paths.append({'P': P_23, 'V': V_23, 'type': '등압 팽창', 'step': '2→3'})

    # 3→4: 단열 팽창
    P_34, V_34 = generate_adiabatic_path(P3, V3, P4, V1, num_points, gas_type)
    paths.append({'P': P_34, 'V': V_34, 'type': '단열 팽창', 'step': '3→4'})

    # 4→1: 등적 방열
    P_41, V_41 = generate_isochoric_path(V1, P4, P1, num_points)
    paths.append({'P': P_41, 'V': V_41, 'type': '등적 방열', 'step': '4→1'})

    # 효율 계산
    efficiency = 1 - (1 / (compression_ratio ** (gamma - 1))) * \
                 ((cutoff_ratio ** gamma - 1) / (gamma * (cutoff_ratio - 1)))

    # 일 계산
    Q_in = mol * Cp * (T3 - T2)
    Q_out = mol * Cv * (T4 - T1)
    W_net = Q_in - abs(Q_out)

    return {
        'paths': paths,
        'states': {
            '1': {'P': P1, 'V': V1, 'T': T1},
            '2': {'P': P2, 'V': V2, 'T': T2},
            '3': {'P': P3, 'V': V3, 'T': T3},
            '4': {'P': P4, 'V': V1, 'T': T4}
        },
        'efficiency': efficiency * 100,
        'W_net': W_net,
        'Q_in': Q_in,
        'Q_out': abs(Q_out),
        'compression_ratio': compression_ratio,
        'cutoff_ratio': cutoff_ratio,
        'cycle_type': 'Diesel'
    }


def generate_brayton_cycle(P1: float, T1: float, pressure_ratio: float,
                           T3: float, gas_type: str = 'diatomic',
                           num_points: int = 50, mol: float = n) -> Dict:
    """
    브레이턴 사이클 (가스터빈)
    1→2: 단열 압축 (압축기)
    2→3: 등압 가열 (연소실)
    3→4: 단열 팽창 (터빈)
    4→1: 등압 방열 (대기)
    """
    gas = get_gas_properties(gas_type)
    gamma = gas['gamma']
    Cp = gas['Cp']

    # 상태 1
    V1 = calculate_volume(T1, P1, mol)

    # 상태 2: 단열 압축 후
    P2 = P1 * pressure_ratio
    T2 = T1 * (pressure_ratio ** ((gamma - 1) / gamma))
    V2 = calculate_volume(T2, P2, mol)

    # 상태 3: 등압 가열 후 (최고 온도)
    P3 = P2
    V3 = calculate_volume(T3, P3, mol)

    # 상태 4: 단열 팽창 후
    P4 = P1
    T4 = T3 / (pressure_ratio ** ((gamma - 1) / gamma))
    V4 = calculate_volume(T4, P4, mol)

    # 경로 생성
    paths = []

    # 1→2: 단열 압축
    P_12, V_12 = generate_adiabatic_path(P1, V1, P2, V2, num_points, gas_type)
    paths.append({'P': P_12, 'V': V_12, 'type': '단열 압축', 'step': '1→2'})

    # 2→3: 등압 가열
    P_23, V_23 = generate_isobaric_path(P2, V2, V3, num_points)
    paths.append({'P': P_23, 'V': V_23, 'type': '등압 가열', 'step': '2→3'})

    # 3→4: 단열 팽창
    P_34, V_34 = generate_adiabatic_path(P3, V3, P4, V4, num_points, gas_type)
    paths.append({'P': P_34, 'V': V_34, 'type': '단열 팽창', 'step': '3→4'})

    # 4→1: 등압 방열
    P_41, V_41 = generate_isobaric_path(P4, V4, V1, num_points)
    paths.append({'P': P_41, 'V': V_41, 'type': '등압 방열', 'step': '4→1'})

    # 효율 계산
    efficiency = 1 - (1 / (pressure_ratio ** ((gamma - 1) / gamma)))

    # 일 계산
    Q_in = mol * Cp * (T3 - T2)
    Q_out = mol * Cp * (T4 - T1)
    W_net = Q_in - abs(Q_out)

    return {
        'paths': paths,
        'states': {
            '1': {'P': P1, 'V': V1, 'T': T1},
            '2': {'P': P2, 'V': V2, 'T': T2},
            '3': {'P': P3, 'V': V3, 'T': T3},
            '4': {'P': P4, 'V': V4, 'T': T4}
        },
        'efficiency': efficiency * 100,
        'W_net': W_net,
        'Q_in': Q_in,
        'Q_out': abs(Q_out),
        'pressure_ratio': pressure_ratio,
        'cycle_type': 'Brayton'
    }


def generate_carnot_cycle(P1: float, V1: float, T_hot: float, T_cold: float,
                          gas_type: str = 'monatomic', num_points: int = 50,
                          mol: float = n) -> Dict:
    """
    카르노 사이클 (이론적 최대 효율)
    1→2: 등온 팽창 (고온)
    2→3: 단열 팽창
    3→4: 등온 압축 (저온)
    4→1: 단열 압축
    """
    gas = get_gas_properties(gas_type)
    gamma = gas['gamma']

    # 상태 1: 고온 등온선 시작
    T1 = T_hot
    # V1과 P1은 입력받음

    # 상태 2: 고온 등온선 끝, 단열선 시작
    # V2를 적절히 선택 (팽창)
    V2 = V1 * 2
    P2 = (mol * R * T_hot) / V2

    # 상태 3: 저온 등온선 끝
    # 단열 과정: T1*V2^(gamma-1) = T_cold*V3^(gamma-1)
    V3 = V2 * ((T_hot / T_cold) ** (1 / (gamma - 1)))
    P3 = (mol * R * T_cold) / V3

    # 상태 4: 저온 등온선 시작
    V4 = V1 * ((T_hot / T_cold) ** (1 / (gamma - 1)))
    P4 = (mol * R * T_cold) / V4

    # 경로 생성
    paths = []

    # 1→2: 등온 팽창
    P_12, V_12 = generate_isothermal_path(P1, V1, P2, V2, num_points, mol)
    paths.append({'P': P_12, 'V': V_12, 'type': '등온 팽창', 'step': '1→2'})

    # 2→3: 단열 팽창
    P_23, V_23 = generate_adiabatic_path(P2, V2, P3, V3, num_points, gas_type)
    paths.append({'P': P_23, 'V': V_23, 'type': '단열 팽창', 'step': '2→3'})

    # 3→4: 등온 압축
    P_34, V_34 = generate_isothermal_path(P3, V3, P4, V4, num_points, mol)
    paths.append({'P': P_34, 'V': V_34, 'type': '등온 압축', 'step': '3→4'})

    # 4→1: 단열 압축
    P_41, V_41 = generate_adiabatic_path(P4, V4, P1, V1, num_points, gas_type)
    paths.append({'P': P_41, 'V': V_41, 'type': '단열 압축', 'step': '4→1'})

    # 카르노 효율
    efficiency = (1 - T_cold / T_hot) * 100

    # 일 계산
    W_12 = mol * R * T_hot * np.log(V2 / V1)
    W_34 = mol * R * T_cold * np.log(V4 / V3)
    W_net = W_12 + W_34  # W_23과 W_41은 상쇄

    Q_in = mol * R * T_hot * np.log(V2 / V1)
    Q_out = abs(mol * R * T_cold * np.log(V4 / V3))

    return {
        'paths': paths,
        'states': {
            '1': {'P': P1, 'V': V1, 'T': T_hot},
            '2': {'P': P2, 'V': V2, 'T': T_hot},
            '3': {'P': P3, 'V': V3, 'T': T_cold},
            '4': {'P': P4, 'V': V4, 'T': T_cold}
        },
        'efficiency': efficiency,
        'W_net': W_net,
        'Q_in': Q_in,
        'Q_out': Q_out,
        'T_hot': T_hot,
        'T_cold': T_cold,
        'cycle_type': 'Carnot'
    }


# ==================== 경로 속성 계산 ====================

def calculate_path_properties(P_array: np.ndarray, V_array: np.ndarray,
                              path_type: str = "일반", gas_type: str = 'monatomic',
                              mol: float = n) -> Optional[Dict]:
    """경로의 모든 열역학적 성질 계산"""
    if len(P_array) < 2 or len(V_array) < 2:
        return None

    P1, V1 = P_array[0], V_array[0]
    P2, V2 = P_array[-1], V_array[-1]

    # 일 계산
    if path_type == "등온":
        W = calculate_work_isothermal(P1, V1, P2, V2, mol)
    elif path_type == "등압":
        W = calculate_work_isobaric(P1, V1, V2)
    elif path_type == "등적":
        W = calculate_work_isochoric(V1, P1, P2)
    elif path_type == "단열":
        W = calculate_work_adiabatic(P1, V1, P2, V2, gas_type, mol)
    else:
        W = calculate_work_general(P_array, V_array)

    # 내부에너지 변화
    dU = calculate_internal_energy_change(P1, V1, P2, V2, gas_type, mol)

    # 엔탈피 변화
    dH = calculate_enthalpy_change(P1, V1, P2, V2, gas_type, mol)

    # 열
    Q = calculate_heat(dU, W)

    # 엔트로피 변화
    dS = calculate_entropy_change(P1, V1, P2, V2, gas_type, mol)

    # 깁스 자유에너지 변화
    dG = calculate_gibbs_free_energy_change(P1, V1, P2, V2, gas_type, mol)

    # 가역 과정 일 (등온 기준)
    T1 = calculate_temperature(P1, V1, mol)
    if V1 > 0 and V2 > 0:
        W_reversible = mol * R * T1 * np.log(V2 / V1)
    else:
        W_reversible = abs(W)

    # 효율
    efficiency = calculate_efficiency(W, W_reversible)

    return {
        'W': W,
        'Q': Q,
        'dU': dU,
        'dH': dH,
        'dS': dS,
        'dG': dG,
        'W_reversible': W_reversible,
        'efficiency': efficiency,
        'P_array': P_array,
        'V_array': V_array,
        'type': path_type,
        'gas_type': gas_type,
        'T1': T1,
        'T2': calculate_temperature(P2, V2, mol)
    }
