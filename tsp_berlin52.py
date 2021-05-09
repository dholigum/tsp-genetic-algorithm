import csv
import tspga as ga

berlinList = []
index = 0
points = []
path = []

if __name__ == '__main__':
    
    # Open yk11 file dataset
    with open('berlin52.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            berlinList.append(ga.City(x=float(row[0]), y=float(row[1])))
    
    print("==========================")
    print("Opening berlin52 dataset")
    print("==========================")
    
    # Get the best route using GA
    berlin52_tsp = ga.geneticAlgorithm(population=berlinList, popSize=350,
                    eliteSize=60, mutationRate=0.0008, generations=800)
    
    print("=========================")
    print("List of ordered route: ")
    print(berlin52_tsp)
    
    # Plot fitness score graph
    ga.geneticAlgorithmPlot(population=berlinList, popSize=350, eliteSize=60, 
                    mutationRate=0.0008, generations=800, dataset='berlin52')
    
    # Plot map visualization
    for point in berlin52_tsp:
        points.append((point.x,point.y))
        path.append(index)
        index = index + 1
        
    # Run the map visualization function
    ga.plotTSP([path], points, 1, dataset='berlin52')