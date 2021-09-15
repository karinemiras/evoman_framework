
class MutationSelection:
    # This class is used to select individuals to be mutated

    selction_ratio = 0.3

    def only_paretns(parents, offspring, population):
        selected_count = round(MutationSelection.selction_ratio * parents.shape[0])
        return parents[:selected_count, :]

    def only_offspring(parents, offspring, population):
        selected_count = round(MutationSelection.selction_ratio * offspring.shape[0])
        return offspring[:selected_count, :]

    def whole_population(parents, offspring, population):
        selected_count = round(MutationSelection.selction_ratio * population.shape[0])
        return population[:selected_count, :]
