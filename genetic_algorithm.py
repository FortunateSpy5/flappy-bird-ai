from numpy import exp
from random import random, randint, choice
import pygame


class Unit(object):
    """
    Unit represents each unit of the genetic algorithm.
    Properties:-
    1. values: The weights and biases.
    2. bird: The bird object representating each unit.
    3. index: The index of the unit in current generation.
    4. fitness: The fitness score of the unit.
    5. is_winner: Boolean value depicting whether the unit is one of the top units.
    """

    def __init__(self, values, bird, index, fitness=0, is_winner=False):
        self.values = values
        self.bird = bird
        self.index = index
        self.fitness = fitness
        self.is_winner = is_winner


class Bird(object):
    """
    Bird represents the units to be drawn on screen.
    x, y: Coordinates of the bird.
    height, width: Dimensions of the bird.
    velocity: Current velocity.
    max_velocity: Maximum downward velocity that the bird can achieve.
    acceleration: Downward acceleration due to gravity (does not act while jumping/flapping).
    is_jumping: Boolean value telling whether the bird is currently jumping/flapping.
    jump_value: Total height gained by flapping/jumping.
    jump_count: Height gained till the current frame from the start of last flap/jump.
    is_dead: Whether the bird has died or not.
    initial_coordinates: Stores starting coordinate for reset.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.height = 45
        self.width = 55
        self.velocity = 0
        self.max_velocity = 4
        self.acceleration = 0.05
        self.is_jumping = False
        self.jump_value = 75
        self.jump_count = 0
        self.is_dead = False
        self.initial_coordinates = (x, y)

    def reset(self):
        """
        This function is used to reset the values of a bird.
        In a particular generation some of the 'winner' birds are kept for the next generation.
        Some properties of those birds needs to be reset.
        """
        self.x = self.initial_coordinates[0]
        self.y = self.initial_coordinates[1]
        self.velocity = 0
        self.is_jumping = False
        self.jump_count = 0
        self.is_dead = False


class Genetic(object):
    """
    Genetic class performs all the functions of the genetic algorithm.
    max_units: Number of units in a particular generation.
    top units: Number of units in a generation that have a chance of passing on their genes.
    population: List of all units of a particular generation.
    scale: Value for scaling the inputs.
    generation: Current generation.
    mutation_rate: Chance of mutation.
    best_population, best_fitness: Best performing unit's generation and fitness.
    """

    def __init__(self, max_units, top_units):
        self.max_units = max_units
        self.top_units = top_units
        self.population = []
        self.scale = 200
        self.generation = 1
        self.mutation_rate = 1
        self.best_population = 0
        self.best_fitness = 0

    def create_population(self, gene_count, x, y):
        """
        This function is used to create the population for first generation.
        n: Number of genetic information(values)
        x, y: Initial coordinates of the birds.
        Population consists of Unit objects.
        The values (weights and biases) are generated in random.
        """
        self.population = []
        for index in range(self.max_units):
            unit = Unit([self.get_random(1) for _ in range(gene_count)], Bird(x, y), index)
            self.population.append(unit)

    @staticmethod
    def get_random(n, lb=-100, ub=100):
        """
        random_mode: To choose which method to use.
        1: return integer in range [lb, ub].
        2: return float in range [0, 1).
        3: return float in range (-1, 1).
        """
        random_mode = n
        if random_mode == 1:
            return randint(lb, ub)
        elif random_mode == 2:
            return random()
        else:
            n = -1
            while n == -1:
                n = random() * 2 - 1
            return n

    def activate_brain(self, unit, inputs, max_x=350, max_y=350):
        """
        :param unit: Unit of population to evaluate.
        :param inputs: The inputs of the neural network.
        :param max_x: Max value of x input.
        :param max_y: Max value of y input.
        :return: The function returns whether the bird should flap/jump.
        """
        inputs = [self.normalize(inputs[0], max_x) * self.scale, self.normalize(inputs[1], max_y) * self.scale]
        output = self.calculate(unit, inputs)
        if output > 0.5:
            return True
        else:
            return False

    def calculate(self, unit, inputs):
        """
        This function acts as a neural network.
        Since, in this case we know the structure of the neural network, it can be represented easily.
        In values, 0-17 are weights and the rest are biases.
        :param unit: Unit of population to evaluate.
        :param inputs: The inputs of the neural network after processing.
        :return: It returns the final output of the neural network.
        """
        h1 = self.sigmoid(inputs[0] * unit.values[0] + inputs[1] * unit.values[1] + unit.values[18])
        h2 = self.sigmoid(inputs[0] * unit.values[2] + inputs[1] * unit.values[3] + unit.values[19])
        h3 = self.sigmoid(inputs[0] * unit.values[4] + inputs[1] * unit.values[5] + unit.values[20])
        h4 = self.sigmoid(inputs[0] * unit.values[6] + inputs[1] * unit.values[7] + unit.values[21])
        h5 = self.sigmoid(inputs[0] * unit.values[8] + inputs[1] * unit.values[9] + unit.values[22])
        h6 = self.sigmoid(inputs[0] * unit.values[10] + inputs[1] * unit.values[11] + unit.values[23])
        j = h1 * unit.values[12] + h2 * unit.values[13] + h3 * unit.values[14] + h4 * unit.values[15] + h5 * unit.values[16] + h6 * unit.values[17] + unit.values[24]
        return self.sigmoid(j)

    @staticmethod
    def sigmoid(x):
        # Sigmoid activation function
        return 1 / (1 + exp(-x))

    @staticmethod
    def tanh(x):
        # Hyperbolic Tangent activation function
        return 1 - exp(-2 * x) / (1 + exp(-2 * x))

    @staticmethod
    def rectified_linear_units(x):
        # Rectified Linear Units activation function
        if x < 0:
            return 0
        else:
            return x

    def evolve(self, x, y):
        """
        This is the most important function as it deals with the evolution of the units.
        All the top units are retained for the next generation. (40%)
        Some of units are selected at random (may be top unit or not) to be retained. (20%)
        One of the offsprings has the two best units as parents.
        The rest have random top unit parents.
        :return:
        """
        winners = self.selection()
        if winners[0].fitness > self.best_fitness:
            self.best_population = self.generation
            self.best_fitness = winners[0].fitness

        self.mutation_rate = 0.2
        for index in range(self.top_units, self.max_units):
            if index == self.top_units:
                parent_a = winners[0].values
                parent_b = winners[1].values
                offspring = self.crossover(parent_a, parent_b)
                offspring = self.mutation(offspring)
            elif index < int(0.8 * self.max_units):
                parent_a = choice(winners).values
                parent_b = choice(winners).values
                offspring = self.crossover(parent_a, parent_b)
                offspring = self.mutation(offspring)
            else:
                offspring = choice(winners).values
                offspring = self.mutation(offspring)

            unit = Unit(offspring, Bird(x, y), index)
            self.population[index] = unit

        self.generation += 1

    def mutation(self, offspring):
        for i in range(len(offspring)):
            if random() < self.mutation_rate:
                factor = 1 + ((random() - 0.5) * 3 + random() - 0.5)
                offspring[i] = offspring[i] * factor
        return offspring

    @staticmethod
    def crossover(a, b):
        """
        A part of the offspring comes from the a while other part comes from b
        Bias values selected from a random parent
        """
        cut = randint(0, 17)
        return a[0:cut] + b[cut:18] + choice([a, b])[18:]

    def selection(self):
        # Returns population sorted by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for index in range(self.top_units):
            self.population[index].is_winner = True
        return sorted_pop

    @staticmethod
    def normalize(value, max_value):
        # To normalize the inputs
        if value < max_value * (-1):
            value = max_value * (-1)
        elif value > max_value:
            value = max_value

        return value / max_value


class FlappyBird(object):
    """
    FlappyBird uses pygame to self.display the performance of birds and gives the fitness.
    screen_width, screen_height: Dimensions of pygame window.
    window: pygame window.
    font: pygame font
    fps: The Frames-Per-Second of the game (ideal value is 180).
    bg: Fixed background.
    bg2: Moving background.
    bg2_x, bg2_y: Coordinates of bg2.
    column_width, column_height: Dimensions of each column.
    column_dist: Horizontal distance between two columns.
    column_gap: Vertical distance between upper and lower column.
    column_lb, column_ub = Range of y coordinate of upper left corner of bottom column.
    column1, column2: Only two columns are visible on the screen at the same time.
    column, column_r: Images of lower and upper column.
    bird_pic: Image of bird.
    current, next: To keep track of columns.
    """
    def __init__(self):
        self.screen_width = 450
        self.screen_height = 600
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.SysFont('calibri', 20, True, False)
        self.fps = 0
        self.dist = 0
        self.bg = pygame.image.load('assets/bg.png').convert()
        self.bg2 = pygame.image.load('assets/bg2.png').convert()
        self.bg2_y = 529
        self.bg2_x = 0
        self.column_width = 75
        self.column_height = 484
        self.column_dist = 260
        self.column_gap = 150
        self.column_lb = 175
        self.column_ub = 480
        self.column1 = [100, randint(self.column_lb, self.column_ub)]
        self.column2 = [self.column1[0] + self.column_dist, randint(self.column_lb, self.column_ub)]
        self.column = pygame.image.load('assets/pipe.png').convert_alpha()
        self.column = pygame.transform.scale(self.column, (self.column_width, self.column_height))
        self.column_r = pygame.transform.rotate(self.column, 180)
        self.bird_pic = pygame.image.load('assets/bird.png').convert_alpha()
        self.bird_pic = pygame.transform.scale(self.bird_pic, (55, 45))
        self.current = self.column1
        self.next = self.column2
        self.display = True

    def reset(self):
        # To reset values after each generation
        self.dist = 0
        self.column1 = [100, randint(self.column_lb, self.column_ub)]
        self.column2 = [self.column1[0] + self.column_dist, randint(self.column_lb, self.column_ub)]
        self.current = self.column1
        self.next = self.column2

    def play(self, genetics):
        # This function renders all the game graphics and calculates the fitness
        clock = pygame.time.Clock()
        # To keep a track of no. of dead birds in current generation
        deaths = 0

        while deaths < len(genetics.population):
            # FPS Clock
            self.dist = self.dist + 1

            if self.display:
                clock.tick(self.fps)

            # Quit Button
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_1]:
                self.fps = 1
            if keys[pygame.K_2]:
                self.fps = 90
            if keys[pygame.K_3]:
                self.fps = 180
            if keys[pygame.K_4]:
                self.fps = 900
            if keys[pygame.K_5]:
                self.fps = 0
            if keys[pygame.K_F1]:
                self.display = True
            if keys[pygame.K_F2]:
                self.display = False

            # Background
            if self.display:
                self.draw_bg()

            # Columns
            if self.display:
                self.draw_col()

            # If one of the columns is out of screen new column gets generated
            if self.dist >= self.column1[0] + self.screen_width + self.column_width:
                self.column1 = self.column2.copy()
                self.column2 = [self.column1[0] + self.column_dist, randint(self.column_lb, self.column_ub)]
                self.current = self.column1
                self.next = self.column2

            # Bird
            for i in range(len(genetics.population)):
                if not genetics.population[i].bird.is_dead:
                    # Check if bird is performing jump/flap animation
                    if genetics.population[i].bird.is_jumping:
                        if genetics.population[i].bird.jump_count < genetics.population[i].bird.jump_value:
                            genetics.population[i].bird.jump_count += 2
                            genetics.population[i].bird.y -= 2
                            if genetics.population[i].bird.y <= 0:
                                genetics.population[i].bird.y = 0
                                genetics.population[i].bird.is_jumping = False
                                genetics.population[i].bird.jump_count = 0
                        else:
                            # If jump/ flap has been completed
                            genetics.population[i].bird.jump_count = 0
                            genetics.population[i].bird.is_jumping = False
                    else:
                        # Get whether to flap/jump from neural network
                        if genetics.activate_brain(genetics.population[i], [self.x_dist(genetics.population[i].bird), self.y_dist(genetics.population[i].bird)]):
                            genetics.population[i].bird.is_jumping = True
                            genetics.population[i].bird.velocity = 0
                        else:
                            # Update velocity and check for collision with ground
                            if genetics.population[i].bird.velocity < genetics.population[i].bird.max_velocity:
                                genetics.population[i].bird.velocity += genetics.population[i].bird.acceleration
                            if genetics.population[i].bird.y < self.bg2_y - genetics.population[i].bird.height - 4:
                                genetics.population[i].bird.y = genetics.population[i].bird.y + genetics.population[i].bird.velocity
                            else:
                                genetics.population[i].fitness = self.dist + self.x_dist(genetics.population[i].bird)
                                genetics.population[i].bird.is_dead = True
                                deaths += 1

                    if self.display:
                        self.draw_bird(genetics.population[i].bird)

                    # Collision with columns/pipes
                    if self.collision(genetics.population[i].bird):
                        genetics.population[i].fitness = self.dist + self.x_dist(genetics.population[i].bird)
                        genetics.population[i].bird.is_dead = True
                        deaths += 1

                    if self.x_dist(genetics.population[i].bird) < 0:
                        self.current = self.next

            # Texts
            if self.fps == 0:
                fps_text = "Unlimited"
            else:
                fps_text = str(self.fps)
            texts = [f"Generation: {genetics.generation}", f"Best gen: {genetics.best_population}", f"Distance(best): {genetics.best_fitness}", f"Distance(current): {self.dist}", f"Deaths: {deaths}",
                     f"FPS: {fps_text}"]
            if self.display:
                self.draw_text(texts)

            # Update self.display
            if self.display:
                pygame.display.update()

    def draw_bg(self):
        # Draws the background on the screen
        self.window.blit(self.bg, (0, 0))
        self.bg2_x = self.bg2_x - 1
        self.window.blit(self.bg2, (self.bg2_x, self.bg2_y))
        if self.bg2_x + self.bg2.get_rect().width < self.screen_width:
            self.window.blit(self.bg2, (self.bg2_x + self.bg2.get_rect().width, self.bg2_y))
        if self.bg2_x + self.bg2.get_rect().width <= 0:
            self.bg2_x = 0

    def draw_col(self):
        # Draws the columns on the screen
        self.window.blit(self.column, (self.column1[0] - self.dist + self.screen_width, self.column1[1]), (0, 0, self.column_width, self.bg2_y - self.column1[1] - 4))
        self.window.blit(self.column_r, (self.column1[0] - self.dist + self.screen_width, 0), (0, self.column_height - self.column1[1] + self.column_gap, self.column_width, self.column_height))

        self.window.blit(self.column, (self.column2[0] - self.dist + self.screen_width, self.column2[1]), (0, 0, self.column_width, self.bg2_y - self.column2[1] - 4))
        self.window.blit(self.column_r, (self.column2[0] - self.dist + self.screen_width, 0), (0, self.column_height - self.column2[1] + self.column_gap, self.column_width, self.column_height))

    def draw_text(self, texts):
        for index, text in enumerate(texts):
            text_obj = self.font.render(text, True, (0, 0, 0))
            self.window.blit(text_obj, (0, index*20))

    def draw_bird(self, bird):
        # Draws the bird on the screen
        self.window.blit(self.bird_pic, (bird.x, bird.y))

    def x_dist(self, bird):
        # Returns horizontal distance between center of bird and end of current column
        return self.current[0] - self.dist + self.screen_width - bird.x - bird.width / 2 + self.column_width

    def y_dist(self, bird):
        # Returns vertical distance between center of bird and center of column gap
        return self.current[1] - self.column_gap / 2 - bird.y - bird.height / 2

    def collision(self, bird):
        # Checks for collision between bird and current column
        temp1 = self.x_dist(bird)
        temp2 = self.y_dist(bird)
        if temp1 + bird.width / 2 > 0 and temp1 - bird.width / 2 < self.column_width:
            if abs(temp2) > self.column_gap / 2 - bird.height / 2:
                return True
        return False


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Flappy Bird")

    genetics = Genetic(100, 40)
    flappy_bird = FlappyBird()

    genetics.create_population(25, flappy_bird.screen_width // 3, flappy_bird.screen_height // 2)
    flappy_bird.play(genetics)

    genetics.generation = 1

    while genetics.generation < 100:
        print(genetics.generation)
        genetics.evolve(flappy_bird.screen_width // 3, flappy_bird.screen_height // 2)
        for pop in genetics.population:
            pop.bird.reset()
        flappy_bird.reset()
        flappy_bird.play(genetics)

    print(genetics.best_fitness)

    pygame.quit()
