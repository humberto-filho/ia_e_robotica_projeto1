import heapq
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt


class AStarPathfinder:
    def __init__(self, map_array: np.array, start: tuple, goal: tuple,
                 wall_influence=5.0, buffer_factor=2.0):
        """Inicializa o planejador A*."""
        self.start = start
        self.goal = goal
        self.wall_influence = float(wall_influence)
        self.buffer_factor = float(buffer_factor)
        self.GOAL_REACHEABLE = False

        # Mantém uma cópia para visualização e outra normalizada para a busca.
        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)

        # Distância até obstáculos e custo adicional de proximidade de paredes.
        self.obstacle_distance = distance_transform_edt(self.map_array != 0)
        self.potential_field = self.create_potential_field()

        # Margem mínima rígida de 1 pixel: evita raspar/cortar quinas de paredes.
        # A distância maior é tratada de forma suave pelo campo potencial.
        self.min_clearance = 1.0

    def preprocess_map(self, map_array: np.array) -> np.array:
        """
        Normaliza o mapa para três estados:
          0   -> obstáculo
          128 -> desconhecido
          255 -> livre
        """
        array = np.asarray(map_array).copy().astype(np.uint8)
        processed = np.full(array.shape, 128, dtype=np.uint8)

        processed[array <= 60] = 0
        processed[array >= 250] = 255
        processed[(array > 60) & (array < 250)] = 128

        return processed

    def create_potential_field(self) -> np.array:
        """
        Cria um campo de custo que cresce perto das paredes.

        Obstáculos recebem custo infinito. Nas demais células, o custo cai
        exponencialmente conforme aumenta a distância até o obstáculo.
        """
        obstacles = self.map_array == 0
        distances = distance_transform_edt(~obstacles)

        scale = max(self.buffer_factor, 1e-6)
        field = self.wall_influence * np.exp(-distances / scale)
        field[obstacles] = np.inf

        return field

    def heuristic(self, a: tuple, b: tuple) -> float:
        """Distância euclidiana entre duas células."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _inside_map(self, point: tuple) -> bool:
        row, col = point
        rows, cols = self.map_array.shape
        return 0 <= row < rows and 0 <= col < cols

    def _walkable(self, point: tuple) -> bool:
        """Retorna False somente para células fora do mapa ou obstáculos."""
        return self._inside_map(point) and self.map_array[point] != 0

    def _safe_known_cell(self, point: tuple) -> bool:
        """Verifica se uma célula conhecida é livre e tem margem da parede."""
        return (
            self._inside_map(point)
            and self.map_array[point] == 255
            and self.obstacle_distance[point] > self.min_clearance
        )

    def find_path(self):
        """
        Executa A* até o objetivo.

        A busca pode projetar uma rota por região desconhecida, mas o robô
        nunca recebe essa parte: know_path() corta a rota na última célula
        conhecida antes da fronteira. Assim ele anda, percebe mais ambiente e
        pode planejar novamente com o mapa atualizado.
        """
        if not self._walkable(self.start):
            print("Ponto inicial inválido ou sobre um obstáculo.")
            return None, None

        if not self._walkable(self.goal):
            print("Objetivo inválido ou sobre um obstáculo.")
            return None, None

        # 8-conectividade produz caminhos menos quadriculados.
        sqrt2 = math.sqrt(2.0)
        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, sqrt2),
            (-1, 1, sqrt2),
            (1, -1, sqrt2),
            (1, 1, sqrt2),
        ]

        open_heap = []
        counter = 0
        heapq.heappush(
            open_heap,
            (self.heuristic(self.start, self.goal), counter, self.start),
        )

        came_from = {}
        g_score = {self.start: 0.0}
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current == self.goal:
                self.GOAL_REACHEABLE = True
                return came_from, current

            closed.add(current)
            row, col = current

            for drow, dcol, movement_cost in moves:
                neighbor = (row + drow, col + dcol)

                if not self._walkable(neighbor) or neighbor in closed:
                    continue

                # Em região conhecida, exige uma pequena margem física.
                if self.map_array[neighbor] == 255:
                    if self.obstacle_distance[neighbor] <= self.min_clearance:
                        continue

                # Não permite que um movimento diagonal corte a quina de uma
                # parede. As duas células ortogonais adjacentes devem existir
                # e não podem ser obstáculos.
                if drow != 0 and dcol != 0:
                    side_a = (row + drow, col)
                    side_b = (row, col + dcol)
                    if not self._walkable(side_a) or not self._walkable(side_b):
                        continue

                # Campo potencial só influencia células já conhecidas.
                # O desconhecido recebe uma pequena penalidade para que, quando
                # houver opção conhecida equivalente, ela seja preferida.
                if self.map_array[neighbor] == 255:
                    wall_cost = float(self.potential_field[neighbor])
                    unknown_cost = 0.0
                else:
                    wall_cost = 0.0
                    unknown_cost = 1.0

                tentative_g = (
                    g_score[current]
                    + movement_cost
                    + wall_cost
                    + unknown_cost
                )

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    counter += 1
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        print("Caminho não encontrado")
        return None, None

    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """Reconstrói start -> ... -> current usando os predecessores."""
        path = [current]

        while current != self.start:
            if current not in came_from:
                return []
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def know_path(self, path: list) -> list:
        """
        Mantém somente o prefixo conhecido do caminho.

        Ao encontrar a primeira célula desconhecida (128), retorna até a última
        célula livre conhecida. Esse é o ponto onde o robô deve parar, perceber
        mais do ambiente e então executar um novo planejamento.
        """
        if not path:
            self.GOAL_REACHEABLE = False
            return []

        known_path = []

        for point in path:
            if not self._inside_map(point):
                break
            if self.map_array[point] != 255:
                break
            known_path.append(point)

        self.GOAL_REACHEABLE = bool(
            known_path and known_path[-1] == self.goal
        )

        return known_path

    def _line_is_safe(self, start: tuple, end: tuple) -> bool:
        """
        Testa o segmento entre dois waypoints.

        A amostragem em resolução de pixel impede que a simplificação crie um
        atalho por parede, região desconhecida ou por cima de uma quina.
        """
        row0, col0 = start
        row1, col1 = end
        steps = max(abs(row1 - row0), abs(col1 - col0))

        if steps == 0:
            return self._safe_known_cell(start)

        previous = start

        for i in range(steps + 1):
            t = i / steps
            row = int(round(row0 + (row1 - row0) * t))
            col = int(round(col0 + (col1 - col0) * t))
            point = (row, col)

            if not self._safe_known_cell(point):
                return False

            # Também protege contra corte diagonal de quinas durante o atalho.
            drow = point[0] - previous[0]
            dcol = point[1] - previous[1]
            if drow != 0 and dcol != 0:
                side_a = (previous[0] + drow, previous[1])
                side_b = (previous[0], previous[1] + dcol)
                if not self._safe_known_cell(side_a) or not self._safe_known_cell(side_b):
                    return False

            previous = point

        return True

    def simplify_path(self, path: list) -> list:
        """
        Reduz o número de waypoints sem atravessar paredes/desconhecido.

        Para cada waypoint, procura o ponto mais distante do caminho original
        que ainda possui linha de visada segura. Isso reduz curvas e deixa a
        trajetória mais suave para o robô.
        """
        if not path:
            return []
        if len(path) <= 2:
            return path.copy()

        simplified = [path[0]]
        current_index = 0

        while current_index < len(path) - 1:
            candidate_index = len(path) - 1

            while candidate_index > current_index + 1:
                if self._line_is_safe(
                    path[current_index], path[candidate_index]
                ):
                    break
                candidate_index -= 1

            # Garante progresso mesmo quando nenhum atalho é possível.
            if candidate_index == current_index:
                candidate_index += 1

            simplified.append(path[candidate_index])
            current_index = candidate_index

        return simplified

    def plot_path(self, path: list, simplified_path: list):
        """Exibe mapa, caminho completo e caminho simplificado."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap="gray", vmin=0, vmax=255)
        plt.scatter(
            self.start[1], self.start[0], color="green", s=100, label="Início"
        )
        plt.scatter(
            self.goal[1], self.goal[0], color="blue", s=100, label="Objetivo"
        )

        if path:
            path_x, path_y = zip(*path)
            plt.plot(
                path_y,
                path_x,
                color="magenta",
                linewidth=1,
                label="Caminho Completo",
            )

            if simplified_path:
                simp_x, simp_y = zip(*simplified_path)
                plt.plot(
                    simp_y,
                    simp_x,
                    color="red",
                    linewidth=2,
                    linestyle="--",
                    marker="o",
                    markersize=4,
                    label="Caminho Simplificado",
                )
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis("equal")
        plt.show()

    def run(self, show_path=True):
        """Executa busca, reconstrução, corte do desconhecido e simplificação."""
        print("Iniciando busca pelo caminho...")
        came_from, final_node = self.find_path()

        if final_node is not None:
            print("Reconstruindo caminho...")
            path = self.reconstruct_path(came_from, final_node)

            print("Robô não anda no desconhecido...")
            path = self.know_path(path)

            if not path:
                print("Não existe trecho conhecido seguro para executar.")
                return None

            print("Caminho encontrado, simplificando...")
            simplified_path = self.simplify_path(path)

            if self.GOAL_REACHEABLE:
                print("Objetivo alcançável com o mapa conhecido.")
            else:
                print("Caminho termina na fronteira conhecida; é necessário replanejar após nova percepção.")

            if show_path:
                print("Plotando o caminho...")
                self.plot_path(path, simplified_path)

            return simplified_path

        print("Nenhum caminho pôde ser encontrado.")
        return None


def prep_map(map_path: str) -> np.array:
    """Carrega o PGM e converte para o mapa de ocupação usado no projeto."""
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

    if map_array is None:
        raise FileNotFoundError(f"Não foi possível abrir o mapa: {map_path}")

    map_array[map_array == 0] = 0
    map_array[map_array == 205] = 128
    map_array[map_array == 254] = 255
    map_array[
        (map_array >= 60) & (map_array != 128) & (map_array != 255)
    ] = 0

    map_array = map_array.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    map_array = cv2.morphologyEx(map_array, cv2.MORPH_OPEN, kernel)

    # Ajusta o sistema de coordenadas do mapa para o utilizado pelo projeto.
    map_array = np.flipud(map_array)

    # Espaço ainda não conhecido ao redor do mapa recebido.
    map_array = np.pad(
        map_array,
        ((0, 200), (0, 200)),
        "constant",
        constant_values=128,
    )

    return map_array


def main():
    # Teste local fornecido pelo template. Troque map1...map5 para observar
    # como o caminho muda à medida que o ambiente é revelado.
    map_array = prep_map("map4.pgm")

    astar = AStarPathfinder(
        map_array,
        (60, 20),
        (60, 120),
        wall_influence=10.0,
        buffer_factor=3.0,
    )

    path = astar.run(show_path=True)
    print("Caminho simplificado:", path)


if __name__ == "__main__":
    main()