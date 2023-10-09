# Partie Algorithmique Génétique

## MIGA: Multi Island Genetic Algorithm

Pour améliorer la convergence vers un optimum global et la vitesse de convergence, on préférera la variante MIGA à l'algorithmique génétique classique.

## Population initiale

La population initiale est générée de manière aléatoire, bien qu'il soit possible - avec de minimes modifications - d'injecter des solutions approchées dans cette population initiale.

## Mutation

La mutation implémentée consiste à inverser deux éléments aléatoires choisis dans un individu.

## Crossover

4 méthodes de crossover ont été implémentées.
- Crossover sur un point
- Crossover sur deux points
- Crossover uniforme
- Crossover "un sur deux"

## Iteration

Les meilleurs individus sont conservés d'une génération sur l'autre (élitisme).
Ensuite, des crossover prennent place pour croiser les individus restants afin de remplir la prochaine les nouvelles îles.
Les migrations ont lieux entre les îles si besoin.

## Modules utilisés
- `numpy`
- `tqdm`
- `random`
- `matplotlib` to generate graphs
