"""
최적 경로 탐색 모듈 (Enhanced Version)
- Dijkstra 알고리즘
- A* 알고리즘 (휴리스틱 기반)
- 다양한 제약 조건 지원
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional, Callable
from thermodynamics import (
    calculate_work_general,
    calculate_entropy_change,
    calculate_temperature,
    get_gas_properties,
    R, n
)


def create_grid(P_min: float = 1, P_max: float = 10,
                V_min: float = 1, V_max: float = 10,
                grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """P-V 평면을 격자로 나눔"""
    P_grid = np.linspace(P_min, P_max, grid_size)
    V_grid = np.linspace(V_min, V_max, grid_size)
    return P_grid, V_grid


def find_nearest_grid_point(P: float, V: float,
                           P_grid: np.ndarray, V_grid: np.ndarray) -> Tuple[int, int]:
    """주어진 (P, V) 상태에 가장 가까운 격자점 찾기"""
    i = np.argmin(np.abs(P_grid - P))
    j = np.argmin(np.abs(V_grid - V))
    return i, j


def calculate_edge_weight(P1: float, V1: float, P2: float, V2: float,
                         optimization_target: str = 'max_work') -> float:
    """
    두 상태점 사이의 엣지 가중치 계산

    Args:
        optimization_target:
            'max_work' - 최대 일 (음수 변환)
            'min_entropy' - 최소 엔트로피 생성
            'max_efficiency' - 최대 효율
    """
    P_avg = (P1 + P2) / 2
    dV = V2 - V1
    W = P_avg * dV

    if optimization_target == 'max_work':
        return -W  # 음수로 변환 (최대 일 → 최소 비용)
    elif optimization_target == 'min_entropy':
        dS = calculate_entropy_change(P1, V1, P2, V2)
        return abs(dS) if dS > 0 else 0.01
    elif optimization_target == 'max_efficiency':
        T1 = calculate_temperature(P1, V1)
        if T1 > 0 and dV != 0:
            W_rev = n * R * T1 * np.log(V2 / V1) if V2 > 0 and V1 > 0 else 0
            if W_rev != 0:
                efficiency = W / W_rev
                return -efficiency
        return 0
    else:
        return -W


def is_valid_edge(P1: float, V1: float, P2: float, V2: float,
                 check_entropy: bool = True,
                 constraints: Optional[Dict] = None) -> bool:
    """
    엣지가 유효한지 확인

    Args:
        check_entropy: 엔트로피 체크 여부
        constraints: 추가 제약 조건
            - max_temperature: 최대 온도 제한
            - min_pressure: 최소 압력 제한
            - isothermal_only: 등온 과정만 허용
    """
    # 기본 제약: 양수 값만
    if V2 < 0.1 or P2 < 0.1:
        return False

    # 추가 제약 조건
    if constraints:
        T2 = calculate_temperature(P2, V2)

        if 'max_temperature' in constraints:
            if T2 > constraints['max_temperature']:
                return False

        if 'min_pressure' in constraints:
            if P2 < constraints['min_pressure']:
                return False

        if 'max_pressure' in constraints:
            if P2 > constraints['max_pressure']:
                return False

        if 'isothermal_only' in constraints and constraints['isothermal_only']:
            T1 = calculate_temperature(P1, V1)
            if abs(T2 - T1) > 10:  # 온도 차이 10K 이내
                return False

    # 엔트로피 체크
    if check_entropy:
        dS = calculate_entropy_change(P1, V1, P2, V2)
        if dS < -0.1:
            return False

    return True


def build_graph(P_grid: np.ndarray, V_grid: np.ndarray,
               allow_diagonal: bool = True,
               optimization_target: str = 'max_work',
               constraints: Optional[Dict] = None) -> Dict:
    """격자 그래프 생성"""
    graph = {}
    grid_size = len(P_grid)

    # 8방향 인접 노드 (대각선 포함)
    if allow_diagonal:
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(grid_size):
        for j in range(grid_size):
            node = (i, j)
            graph[node] = []

            P1 = P_grid[i]
            V1 = V_grid[j]

            for di, dj in directions:
                ni, nj = i + di, j + dj

                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    P2 = P_grid[ni]
                    V2 = V_grid[nj]

                    if is_valid_edge(P1, V1, P2, V2, constraints=constraints):
                        weight = calculate_edge_weight(P1, V1, P2, V2, optimization_target)
                        graph[node].append(((ni, nj), weight))

    return graph


# ==================== Dijkstra 알고리즘 ====================

def dijkstra(graph: Dict, start: Tuple[int, int],
            end: Tuple[int, int]) -> Tuple[Optional[List], float]:
    """Dijkstra 알고리즘으로 최단(최적) 경로 찾기"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}

    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == end:
            break

        for neighbor, weight in graph.get(current, []):
            if neighbor in visited:
                continue

            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    # 경로 재구성
    path = []
    current = end

    if previous[current] is None and current != start:
        return None, float('inf')

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()
    return path, distances[end]


# ==================== A* 알고리즘 ====================

def manhattan_heuristic(node: Tuple[int, int], goal: Tuple[int, int],
                       P_grid: np.ndarray, V_grid: np.ndarray) -> float:
    """맨해튼 거리 휴리스틱"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def euclidean_heuristic(node: Tuple[int, int], goal: Tuple[int, int],
                       P_grid: np.ndarray, V_grid: np.ndarray) -> float:
    """유클리드 거리 휴리스틱"""
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)


def thermodynamic_heuristic(node: Tuple[int, int], goal: Tuple[int, int],
                           P_grid: np.ndarray, V_grid: np.ndarray) -> float:
    """
    열역학적 휴리스틱
    등온 과정의 이론적 최대 일을 기반으로 추정
    """
    P1, V1 = P_grid[node[0]], V_grid[node[1]]
    P2, V2 = P_grid[goal[0]], V_grid[goal[1]]

    T = calculate_temperature(P1, V1)

    if V1 > 0 and V2 > 0 and T > 0:
        # 등온 과정의 최대 일 (음수로 변환)
        W_max = -n * R * T * np.log(V2 / V1)
        return W_max * 0.8  # 보수적 추정

    return euclidean_heuristic(node, goal, P_grid, V_grid)


def astar(graph: Dict, start: Tuple[int, int], end: Tuple[int, int],
         P_grid: np.ndarray, V_grid: np.ndarray,
         heuristic: str = 'thermodynamic') -> Tuple[Optional[List], float]:
    """
    A* 알고리즘으로 최적 경로 찾기

    Args:
        heuristic: 휴리스틱 함수 선택
            'manhattan' - 맨해튼 거리
            'euclidean' - 유클리드 거리
            'thermodynamic' - 열역학적 휴리스틱
    """
    # 휴리스틱 함수 선택
    if heuristic == 'manhattan':
        h_func = manhattan_heuristic
    elif heuristic == 'euclidean':
        h_func = euclidean_heuristic
    else:
        h_func = thermodynamic_heuristic

    # g(n): 시작점에서 현재 노드까지의 실제 비용
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    # f(n) = g(n) + h(n)
    f_score = {node: float('inf') for node in graph}
    f_score[start] = h_func(start, end, P_grid, V_grid)

    # 이전 노드 추적
    came_from = {}

    # 우선순위 큐: (f_score, g_score, node)
    # g_score를 추가하여 타이브레이킹
    open_set = [(f_score[start], 0, start)]
    open_set_hash = {start}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current not in open_set_hash:
            continue
        open_set_hash.discard(current)

        if current == end:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[end]

        for neighbor, weight in graph.get(current, []):
            tentative_g = g_score[current] + weight

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + h_func(neighbor, end, P_grid, V_grid)
                f_score[neighbor] = f

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    open_set_hash.add(neighbor)

    return None, float('inf')


# ==================== 메인 함수들 ====================

def find_optimal_path(P1: float, V1: float, P2: float, V2: float,
                     grid_size: int = 50,
                     allow_diagonal: bool = True,
                     algorithm: str = 'dijkstra',
                     optimization_target: str = 'max_work',
                     heuristic: str = 'thermodynamic',
                     constraints: Optional[Dict] = None,
                     gas_type: str = 'monatomic') -> Optional[Dict]:
    """
    최적 경로 찾기 (메인 함수)

    Args:
        algorithm: 'dijkstra' 또는 'astar'
        optimization_target: 'max_work', 'min_entropy', 'max_efficiency'
        heuristic: A* 알고리즘용 휴리스틱
        constraints: 추가 제약 조건
        gas_type: 기체 타입
    """
    # 격자 생성
    P_grid, V_grid = create_grid(grid_size=grid_size)

    # 시작/끝 격자점 찾기
    start_i, start_j = find_nearest_grid_point(P1, V1, P_grid, V_grid)
    end_i, end_j = find_nearest_grid_point(P2, V2, P_grid, V_grid)

    start_node = (start_i, start_j)
    end_node = (end_i, end_j)

    # 그래프 생성
    graph = build_graph(P_grid, V_grid,
                       allow_diagonal=allow_diagonal,
                       optimization_target=optimization_target,
                       constraints=constraints)

    # 알고리즘 실행
    if algorithm == 'astar':
        path_nodes, total_cost = astar(graph, start_node, end_node,
                                       P_grid, V_grid, heuristic)
    else:
        path_nodes, total_cost = dijkstra(graph, start_node, end_node)

    if path_nodes is None or len(path_nodes) == 0:
        return None

    # 경로를 P, V 배열로 변환
    P_array = np.array([P_grid[i] for i, j in path_nodes])
    V_array = np.array([V_grid[j] for i, j in path_nodes])

    # 일 계산
    W = calculate_work_general(P_array, V_array)

    # 다른 성질들 계산
    from thermodynamics import (
        calculate_internal_energy_change,
        calculate_heat,
        calculate_entropy_change,
        calculate_efficiency,
        calculate_enthalpy_change,
        calculate_gibbs_free_energy_change
    )

    dU = calculate_internal_energy_change(P1, V1, P2, V2, gas_type)
    dH = calculate_enthalpy_change(P1, V1, P2, V2, gas_type)
    Q = calculate_heat(dU, W)
    dS = calculate_entropy_change(P1, V1, P2, V2, gas_type)
    dG = calculate_gibbs_free_energy_change(P1, V1, P2, V2, gas_type)

    # 가역 과정 일
    T1 = calculate_temperature(P1, V1)
    if V1 > 0 and V2 > 0:
        W_reversible = n * R * T1 * np.log(V2 / V1)
    else:
        W_reversible = abs(W)

    efficiency = calculate_efficiency(W, W_reversible)

    return {
        'P_array': P_array,
        'V_array': V_array,
        'W': W,
        'Q': Q,
        'dU': dU,
        'dH': dH,
        'dS': dS,
        'dG': dG,
        'W_reversible': W_reversible,
        'efficiency': efficiency,
        'type': f'최적 경로 ({algorithm.upper()})',
        'algorithm': algorithm,
        'optimization_target': optimization_target,
        'path_nodes': path_nodes,
        'total_cost': total_cost,
        'gas_type': gas_type
    }


def compare_algorithms(P1: float, V1: float, P2: float, V2: float,
                      grid_size: int = 50,
                      gas_type: str = 'monatomic') -> Dict:
    """Dijkstra와 A* 알고리즘 비교"""
    import time

    results = {}

    # Dijkstra
    start_time = time.time()
    dijkstra_result = find_optimal_path(P1, V1, P2, V2, grid_size,
                                        algorithm='dijkstra',
                                        gas_type=gas_type)
    dijkstra_time = time.time() - start_time
    results['dijkstra'] = {
        'result': dijkstra_result,
        'time': dijkstra_time
    }

    # A* with different heuristics
    for heuristic in ['manhattan', 'euclidean', 'thermodynamic']:
        start_time = time.time()
        astar_result = find_optimal_path(P1, V1, P2, V2, grid_size,
                                         algorithm='astar',
                                         heuristic=heuristic,
                                         gas_type=gas_type)
        astar_time = time.time() - start_time
        results[f'astar_{heuristic}'] = {
            'result': astar_result,
            'time': astar_time
        }

    return results


def find_multiple_paths(P1: float, V1: float, P2: float, V2: float,
                       num_paths: int = 3,
                       grid_size: int = 50,
                       gas_type: str = 'monatomic') -> List[Dict]:
    """
    여러 개의 대안 경로 찾기
    K-shortest paths 변형 알고리즘
    """
    paths = []

    # 첫 번째 최적 경로
    best_path = find_optimal_path(P1, V1, P2, V2, grid_size,
                                  gas_type=gas_type)
    if best_path:
        paths.append(best_path)

    # 다른 최적화 목표로 경로 찾기
    for target in ['min_entropy', 'max_efficiency']:
        if len(paths) >= num_paths:
            break

        alt_path = find_optimal_path(P1, V1, P2, V2, grid_size,
                                     optimization_target=target,
                                     gas_type=gas_type)
        if alt_path and alt_path not in paths:
            paths.append(alt_path)

    # A* 다른 휴리스틱으로
    for heuristic in ['manhattan', 'euclidean']:
        if len(paths) >= num_paths:
            break

        alt_path = find_optimal_path(P1, V1, P2, V2, grid_size,
                                     algorithm='astar',
                                     heuristic=heuristic,
                                     gas_type=gas_type)
        if alt_path:
            paths.append(alt_path)

    return paths[:num_paths]
