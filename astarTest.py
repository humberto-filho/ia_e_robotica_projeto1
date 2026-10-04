import heapq
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import math

# Convenção do mapa DEPOIS do prep_map:
#   255 = célula LIVRE e CONHECIDA (o robô pode andar)
#   128 = célula DESCONHECIDA (o robô ainda não mapeou — não pode andar, mas o
#         planejador pode atravessá-la para indicar a direção do objetivo)
#     0 = PAREDE / obstáculo (jamais atravessar)
LIVRE = 255
DESCONHECIDO = 128
PAREDE = 0


class AStarPathfinder:
    def __init__(self, map_array: np.array, start: tuple, goal: tuple, wall_influence=5.0, buffer_factor=2.0,
                 map_resolution=0.05, safety_margin_m=0.24):
        """
        Inicializa o A* com mapa, ponto inicial, objetivo e parâmetros de influência.

        Args:
            map_array (np.array): Mapa binário (obstáculos e caminho livre).
            start (tuple): Ponto inicial (linha, coluna).
            goal (tuple): Ponto objetivo (linha, coluna).
            wall_influence (float): Peso da proximidade das paredes.
            buffer_factor (float): Escala da influência das paredes.
            map_resolution (float): Tamanho de cada célula do mapa, em metros
                (padrão 0.05 m = 5 cm, típico dos mapas do ROS).
            safety_margin_m (float): DISTÂNCIA SEGURA CRÍTICA, em metros, do
                centro do robô até qualquer parede. O A* trata como obstáculo
                toda célula a menos dessa distância de uma parede (inflação de
                obstáculos), então o caminho JAMAIS passa mais perto que isso.
        """
        self.start = start
        self.goal = goal
        self.wall_influence = wall_influence
        self.buffer_factor = buffer_factor
        self.GOAL_REACHEABLE = False

        # Distância segura crítica convertida de metros para células.
        # Ex.: 0.24 m / 0.05 m por célula = 4.8 -> 5 células de folga mínima.
        self.map_resolution = map_resolution
        self.safety_margin_m = safety_margin_m
        self.safety_cells = int(math.ceil(safety_margin_m / map_resolution))

        # Distância mínima (em células) exigida na validação do caminho
        # simplificado/suavizado — igual à distância segura crítica.
        self.min_clearance = float(self.safety_cells)

        # Prepara o mapa, expandindo suas bordas e ajustando o array.
        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)

        # Cria um campo potencial baseado no mapa para influenciar o caminho.
        self.potential_field = self.create_potential_field()

    # ------------------------------------------------------------------
    # Pré-processamento e campo potencial
    # ------------------------------------------------------------------

    def preprocess_map(self, map_array: np.array) -> np.array:
        """
        Ajusta o mapa, convertendo valores intermediários para obstáculos.

        Mantém como planejáveis apenas as células LIVRES (255) e DESCONHECIDAS
        (128). Qualquer outro valor (parede = 0 ou tons intermediários) vira
        obstáculo (0), garantindo que o A* nunca atravesse uma parede.

        Args:
            map_array (np.array): Mapa original.

        Returns:
            np.array: Mapa processado.
        """
        processed = map_array.copy()
        # Tudo que não é livre nem desconhecido vira obstáculo.
        processed[(processed != LIVRE) & (processed != DESCONHECIDO)] = PAREDE
        return processed

    def create_potential_field(self) -> np.array:
        """
        Gera campo potencial com base na distância de obstáculos.

        Usa a transformada de distância euclidiana (distance_transform_edt)
        para saber, para cada célula, quão longe ela está da parede mais
        próxima. Quanto mais perto da parede, MAIOR o custo extra de passar
        por ali — é isso que faz o A* preferir caminhos "no meio do corredor".

        A penalidade decai exponencialmente: wall_influence * exp(-d / buffer_factor).
        Como o custo é sempre finito, o A* ainda aceita passar por corredores
        estreitos quando eles são a única opção (exigência da rubrica).

        Returns:
            np.array: Campo potencial.
        """
        walls = (self.map_array == PAREDE)
        # Distância de cada célula até a parede mais próxima (em células).
        self.dist_to_wall = distance_transform_edt(~walls)
        field = self.wall_influence * np.exp(-self.dist_to_wall / self.buffer_factor)
        field[walls] = 0.0  # paredes já são bloqueadas na expansão do A*
        return field

    def heuristic(self, a: tuple, b: tuple) -> float:
        """
        Calcula a heurística entre dois pontos.

        Distância euclidiana (linha reta). É admissível: nunca superestima o
        custo real, pois o caminho verdadeiro é sempre pelo menos tão longo
        quanto a linha reta. Faz o A* priorizar nós que estão mais perto do
        objetivo, reduzindo muito o número de células exploradas.

        Args:
            a (tuple): Ponto A.
            b (tuple): Ponto B.

        Returns:
            float: Resultado da heurística.
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ------------------------------------------------------------------
    # Núcleo do A*
    # ------------------------------------------------------------------

    def _mapa_inflado(self, celulas: float) -> np.array:
        """
        Inflação de obstáculos (configuration space): devolve uma cópia do mapa
        em que toda célula a menos de `celulas` células de uma parede também
        vira PAREDE. É o que garante, de forma RÍGIDA, a distância segura
        crítica: o A* simplesmente não consegue planejar perto da parede.
        """
        inflado = self.map_array.copy()
        inflado[self.dist_to_wall < celulas] = PAREDE
        return inflado

    def find_path(self):
        """
        Executa o algoritmo A* para encontrar caminho até o objetivo.

        Planeja sobre o mapa com obstáculos INFLADOS pela distância segura
        crítica (24 cm do centro do robô): nenhuma célula do caminho fica a
        menos que isso de uma parede. Se não existir caminho com a margem
        completa (corredor estreito demais), tenta de novo com margens
        reduzidas (75%, 50%, 25% e 0%), avisando no console — assim o robô
        não trava em passagens apertadas, mas nunca planeja colado na parede
        por escolha.

        Returns:
            dict: Predecessores dos nós no caminho. Se o caminho não for encontrado, retorna None.
            tuple: O ponto final (objetivo) ou None se não encontrado.
        """
        for fator in (1.0, 0.75, 0.5, 0.25, 0.0):
            inflacao = self.safety_cells * fator
            grade = self._mapa_inflado(inflacao)
            came_from, final_node = self._a_estrela(grade)
            if final_node is not None:
                if fator < 1.0:
                    margem_cm = inflacao * self.map_resolution * 100
                    print(f"Aviso: passagem estreita — margem de segurança reduzida para ~{margem_cm:.0f} cm")
                return came_from, final_node
        print("Caminho não encontrado")
        return None, None

    def _a_estrela(self, grade: np.array):
        """
        Núcleo do A*, operando sobre `grade` (mapa já inflado).

        Estratégia:
          - Fronteira (open list) como min-heap ordenada por f = g + h.
          - Vizinhança 8-conectada (ortogonais custam 1, diagonais custam √2),
            sem "cortar quina" em diagonal entre duas paredes.
          - O custo de entrar numa célula soma a penalidade do campo potencial
            (proximidade de parede) e um pequeno custo extra para células
            desconhecidas, para o robô preferir o que já é conhecido.
          - Células bloqueadas na grade inflada (parede real ou colchão de
            segurança) nunca são expandidas.

        Returns:
            dict: Predecessores dos nós no caminho (ou None).
            tuple: O ponto final (objetivo) ou None se não encontrado.
        """
        start, goal = self.start, self.goal
        rows, cols = grade.shape

        def dentro(r, c):
            return 0 <= r < rows and 0 <= c < cols

        def livre_de_parede(r, c):
            return grade[r, c] != PAREDE

        # Se o objetivo cair numa parede (ou dentro do colchão de segurança),
        # procura a célula transitável mais próxima para ainda assim planejar
        # na direção certa.
        if not dentro(*goal) or not livre_de_parede(*goal):
            goal = self._nearest_free(goal, grade)
            if goal is None:
                return None, None
            self.goal = goal

        # (f, g, célula) — f é o critério de desempate da fila de prioridade.
        open_heap = [(self.heuristic(start, goal), 0.0, start)]
        came_from = {start: None}
        g_score = {start: 0.0}
        closed = set()

        # 8 vizinhos: (dr, dc, custo do passo)
        vizinhos = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                # O objetivo só é "alcançável" de verdade se estiver na
                # região conhecida (livre). Caso contrário, o caminho será
                # truncado na borda do conhecido pelo know_path.
                self.GOAL_REACHEABLE = (self.map[goal[0], goal[1]] == LIVRE)
                return came_from, current

            r, c = current
            for dr, dc, step in vizinhos:
                nr, nc = r + dr, c + dc
                if not dentro(nr, nc) or not livre_de_parede(nr, nc):
                    continue
                # Não corta quina: diagonal só se os dois ortogonais forem livres.
                if dr != 0 and dc != 0:
                    if not livre_de_parede(r + dr, c) or not livre_de_parede(r, c + dc):
                        continue
                cell = self.map_array[nr, nc]
                # Custo = passo + proximidade de parede + leve custo p/ desconhecido.
                custo = step + self.potential_field[nr, nc]
                if cell == DESCONHECIDO:
                    custo += 0.1
                tentative = g + custo
                if tentative < g_score.get((nr, nc), math.inf):
                    g_score[(nr, nc)] = tentative
                    came_from[(nr, nc)] = current
                    f = tentative + self.heuristic((nr, nc), goal)
                    heapq.heappush(open_heap, (f, tentative, (nr, nc)))

        return None, None

    def _nearest_free(self, cell: tuple, grade: np.array):
        """Retorna a célula transitável (na grade inflada) mais próxima de `cell` (ou None)."""
        r0, c0 = int(round(cell[0])), int(round(cell[1]))
        best, best_d = None, math.inf
        free_cells = np.argwhere(grade != PAREDE)
        if len(free_cells) == 0:
            return None
        for r, c in free_cells:
            d = (r - r0) ** 2 + (c - c0) ** 2
            if d < best_d:
                best_d, best = d, (int(r), int(c))
        return best

    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """
        Reconstrói o caminho a partir do ponto final até o inicial.

        Segue a cadeia de predecessores (came_from) do objetivo de volta ao
        início e devolve a lista na ordem início -> fim.

        Args:
            came_from (dict): O dicionário de predecessores no caminho.
            current (tuple): O ponto final (objetivo).

        Returns:
            list: Lista de tuplas com caminho reconstruído.
        """
        path = [current]
        while came_from.get(path[-1]) is not None:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def know_path(self, path: list) -> list:
        """
        Remove trechos desconhecidos e ajusta o caminho, se necessário.

        O robô não anda no desconhecido: percorre o caminho do início e o
        trunca na primeira célula que não é livre e conhecida (255). O robô
        anda até essa borda, percebe mais do ambiente e replaneja.

        Args:
            path (list): Caminho completo.

        Returns:
            list: Caminho ajustado.
        """
        if not path:
            return path
        known = [path[0]]
        for cell in path[1:]:
            if self.map[cell[0], cell[1]] == LIVRE:
                known.append(cell)
            else:
                break
        return known

    # ------------------------------------------------------------------
    # Suavização do caminho (movimentos suaves, sem ziguezague)
    # ------------------------------------------------------------------

    def simplify_path(self, path: list) -> list:
        """
        Simplifica o caminho removendo direções repetidas.

        Três etapas:
          1) Remove pontos colineares (trechos em linha reta viram 2 waypoints).
          2) Suavização por linha de visão ("shortcut"): liga waypoints
             distantes por uma reta SEMPRE que a reta inteira mantiver
             distância segura das paredes (min_clearance). Isso elimina o
             ziguezague célula a célula.
          3) Suavização por corte de cantos (algoritmo de Chaikin): cada
             canto vivo do caminho é substituído por uma curva suave, mas
             SOMENTE se a curva mantiver clearance das paredes — em
             corredores estreitos o canto é preservado.

          O resultado são poucos segmentos longos e curvas gentis — o tipo
          de referência que um controlador (como o PID do vídeo de controle
          de carros autônomos) segue sem movimentos abruptos.

        Args:
            path (list): Caminho completo.

        Returns:
            list: Caminho simplificado e suavizado.
        """
        if not path or len(path) < 3:
            return path

        # Etapa 1: remove pontos colineares (mesma direção do anterior).
        reduzido = [path[0]]
        for i in range(1, len(path) - 1):
            d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if d1 != d2:
                reduzido.append(path[i])
        reduzido.append(path[-1])

        # Etapa 2: shortcut por linha de visão com verificação de clearance.
        suave = [reduzido[0]]
        anchor = 0
        while anchor < len(reduzido) - 1:
            proximo = anchor + 1
            # Tenta ligar o anchor ao waypoint mais distante possível.
            for cand in range(len(reduzido) - 1, anchor, -1):
                if self._line_of_sight(reduzido[anchor], reduzido[cand]):
                    proximo = cand
                    break
            suave.append(reduzido[proximo])
            anchor = proximo

        # Etapa 3: corta os cantos vivos, gerando curvas suaves e seguras.
        return self._chaikin_seguro(suave, iteracoes=3)

    def _chaikin_seguro(self, pontos: list, iteracoes=3) -> list:
        """
        Suavização por corte de cantos (variante do algoritmo de Chaikin).

        A cada iteração, cada vértice (canto vivo) é substituído por dois
        pontos posicionados a 25% dos segmentos vizinhos — arredondando o
        canto. O corte só é aceito se os novos trechos não se aproximarem
        demais das paredes; caso contrário o canto original é mantido.
        Repetir algumas iterações produz uma curva suave (semelhante a uma
        B-spline) que nunca sacrifica a segurança.
        """
        pts = [tuple(map(float, p)) for p in pontos]
        for _ in range(iteracoes):
            if len(pts) < 3:
                break
            novo = [pts[0]]
            for i in range(1, len(pts) - 1):
                a, v, b = novo[-1], pts[i], pts[i + 1]
                q1 = (0.75 * v[0] + 0.25 * a[0], 0.75 * v[1] + 0.25 * a[1])
                q2 = (0.75 * v[0] + 0.25 * b[0], 0.75 * v[1] + 0.25 * b[1])
                if self._segmento_seguro(a, q1) and self._segmento_seguro(q1, q2):
                    novo.extend([q1, q2])   # canto vira curva
                else:
                    novo.append(v)          # canto preservado (espaço estreito)
            novo.append(pts[-1])
            pts = novo
        return pts

    def _segmento_seguro(self, a: tuple, b: tuple) -> bool:
        """
        Verifica se o segmento a->b é seguro: não atravessa parede e mantém
        pelo menos min_clearance células de distância das paredes.
        Funciona com coordenadas inteiras ou fracionárias.
        """
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(dist * 3) + 1)
        for t in np.linspace(0, 1, n):
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if self.map_array[r, c] == PAREDE:
                return False
            if self.dist_to_wall[r, c] < self.min_clearance:
                return False
        return True

    def _line_of_sight(self, a: tuple, b: tuple) -> bool:
        """
        Verifica se a reta entre a e b é segura (mesma regra do
        _segmento_seguro): sem parede e com clearance mínimo.
        """
        return self._segmento_seguro(a, b)

    # ------------------------------------------------------------------
    # Visualização e orquestração
    # ------------------------------------------------------------------

    def path_to_xy(self, path: list) -> list:
        """
        Converte o caminho da convenção interna (linha, coluna) para
        (x, y) = (coluna, linha), com coordenadas arredondadas em 2 casas.

        Args:
            path (list): Caminho em (linha, coluna).

        Returns:
            list: Caminho como lista de tuplas [(x1, y1), (x2, y2), ...].
        """
        return [(round(float(c), 2), round(float(r), 2)) for r, c in path]

    def plot_path(self, path: list, simplified_path: list, save_path: str = None):
        """
        Exibe o mapa com o caminho completo e o simplificado.

        Args:
            path (list): O caminho completo encontrado.
            simplified_path (list): O caminho simplificado encontrado.
            save_path (str): Se informado, salva o gráfico nesse arquivo (ex.: 'caminho.png').
        """
        simplified_path = self.simplify_path(path)

        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='gray')
        plt.scatter(self.start[1], self.start[0], color='green', s=100, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=100, label='Objetivo')

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='magenta', linewidth=1, label='Caminho Completo')
            simp_x, simp_y = zip(*simplified_path)
            plt.plot(simp_y, simp_x, color='red', linewidth=2, linestyle='--', label='Caminho Simplificado')
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis('equal')
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        plt.show()

    def run(self, show_path=True, save_path=None):
        """
        Essa função é chamada pelo navegador para executar o algoritmo A* e gerar o caminho.
        Executa o processo completo: busca, reconstrução, simplificação e visualização do caminho.

        Args:
            show_path (bool): Se True, exibe o caminho graficamente.
            save_path (str): Se informado, salva o gráfico do caminho nesse arquivo.

        Returns:
            list or None: Caminho simplificado (em (linha, coluna)) ou None se não encontrado.
            O mesmo caminho em (x, y) fica disponível em self.caminho_xy.
        """
        print("Iniciando busca pelo caminho...")
        came_from, final_node = self.find_path()

        if final_node:
            print("Reconstruindo caminho...")
            path = self.reconstruct_path(came_from, final_node)

            print("Robo não anda no disconhecido")
            path = self.know_path(path)

            print("Caminho encontrado, simplificando...")
            simplified_path = self.simplify_path(path)

            # Entrega o caminho simplificado como lista [(x1, y1), (x2, y2), ...]
            self.caminho_xy = self.path_to_xy(simplified_path)
            print(f"Caminho simplificado ({len(self.caminho_xy)} pontos, formato (x, y)):")
            print(self.caminho_xy)

            print("Plotando o caminho...")
            if show_path:
                self.plot_path(path, simplified_path, save_path=save_path)

            return simplified_path
        else:
            print("Nenhum caminho pôde ser encontrado.")
            self.caminho_xy = None
            return None


def prep_map(map_path: str) -> np.array:
    """
    Prepara o mapa carregando e processando a imagem de entrada.

    Args:
        map_path (str): O caminho do arquivo do mapa.

    Returns:
        np.array: O mapa processado como um array numpy.
    """
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map_array[map_array == 0] = 0
    map_array[map_array == 205] = 128
    map_array[map_array == 254] = 255
    map_array[(map_array >= 60) & (map_array != 128) & (map_array != 255)] = 0
    map_array = map_array.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    map_array = cv2.morphologyEx(map_array, cv2.MORPH_OPEN, kernel)
    map_array = np.flipud(map_array)
    map_array = np.pad(map_array, ((0, 200), (0, 200)), 'constant', constant_values=128)
    return map_array


def main():
    # Testa o planejador nos 5 mapas de exemplo.
    for i in range(1, 6):
        print(f"\n===== map{i}.pgm =====")
        map_array = prep_map(f'map{i}.pgm')
        astar = AStarPathfinder(map_array, (60, 20), (60, 120),
                                wall_influence=10.0, buffer_factor=3.0)
        # Ao rodar: plota o gráfico (e salva em caminho_map{i}.png) e imprime
        # a lista [(x1, y1), (x2, y2), ...] com o caminho simplificado.
        caminho = astar.run(save_path=f'caminho_map{i}.png')
        if caminho:
            print(f"Objetivo alcançável (região conhecida): {astar.GOAL_REACHEABLE}")


if __name__ == '__main__':
    main()