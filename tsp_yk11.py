import csv
import tspga as ga

ykList = []
index = 0
points = []
path = []

if __name__ == '__main__':
    
    # Open yk11 file dataset
    with open('yk11.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ykList.append(ga.City(x=float(row[0]), y=float(row[1])))
    
    print("==========================")
    print("Opening yk11 dataset")
    print("==========================")
    
    # Get the best route using GA
    yk11_tsp = ga.geneticAlgorithm(population=ykList, popSize=100, 
                    eliteSize=20, mutationRate=0.01, generations=60)
    
    print("=========================")
    print("List of ordered route: ")
    print(yk11_tsp)
    
    # Plot fitness score graph
    ga.geneticAlgorithmPlot(population=ykList, popSize=100, eliteSize=20, 
                                mutationRate=0.01, generations=60, dataset='yk11')
    
    # Plot map visualization
    for point in yk11_tsp:
        points.append((point.x,point.y))
        path.append(index)
        index = index + 1
        
    # Run the map visualization function
    ga.plotTSP([path], points, 1, dataset='yk11')