import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch_geometric.data import DataLoader

class GraphVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def draw_combined_node_properties(self, save_dir=None):
        # Inicializar una lista para acumular los datos de cada columna
        combined_node_attrs = []

        # Iterar sobre el conjunto de datos y acumular los datos de cada columna
        for pyg_graph in self.dataset:
            node_attrs = pyg_graph.x.numpy()
            combined_node_attrs.append(node_attrs)

        # Convertir la lista acumulada en un array numpy
        combined_node_attrs = np.vstack(combined_node_attrs)

        # Crear el directorio si no existe
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Dibujar cada columna del tensor x acumulado por separado
        for i in range(combined_node_attrs.shape[1]):
            plt.figure(figsize=(10, 6))
            plt.hist(combined_node_attrs[:, i], bins=50, color='b', alpha=0.7)
            plt.xlabel(f'Propiedad {i}')
            plt.ylabel(f'Events')
            plt.title(f'Propiedad {i} de todos los gráficos')
            
            # Guardar o mostrar el gráfico
            if save_dir:
                plt.savefig(f"{save_dir}/combined_property_{i}.png")
            else:
                plt.show()

    def draw_combined_edge_properties(self, save_dir=None):
        # Inicializar una lista para acumular los datos de cada columna
        combined_edge_deltaPhi = []
        combined_edge_deltaEta = [] 

        # Iterar sobre el conjunto de datos y acumular los datos de cada columna
        for pyg_graph in self.dataset:
            combined_edge_deltaPhi.append(pyg_graph.deltaPhi.numpy().flatten())
            combined_edge_deltaEta.append(pyg_graph.deltaEta.numpy().flatten())

        combined_edge_deltaPhi = np.concatenate(combined_edge_deltaPhi)
        combined_edge_deltaEta = np.concatenate(combined_edge_deltaEta)

        # Crear el directorio si no existe
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(10, 6))
        plt.hist(combined_edge_deltaPhi, bins=50, range=(-500,500), color='r', alpha=0.7)
        plt.xlabel(f'DeltaPhi')
        plt.ylabel(f'Events')
            
        # Guardar o mostrar el gráfico
        if save_dir:
            plt.savefig(f"{save_dir}/combined_edge_property_deltaPhi.png")
        else:
            plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.hist(combined_edge_deltaEta, bins=50, range=(-500,500), color='r', alpha=0.7)
        plt.xlabel(f'DeltaEta')
        plt.ylabel(f'Events')
          
        # Guardar o mostrar el gráfico
        if save_dir:
            plt.savefig(f"{save_dir}/combined_edge_property_deltaEta.png")
        else:
            plt.show()


    def draw_combined_node_y(self, save_dir=None):
        # Inicializar una lista para acumular los datos de y
        combined_node_y = []

        # Iterar sobre el conjunto de datos y acumular los datos de y
        for pyg_graph in self.dataset:
            node_y = pyg_graph.y.numpy()
            combined_node_y.append(node_y)

        # Convertir la lista acumulada en un array numpy
        combined_node_y = np.vstack(combined_node_y)

        # Crear el directorio si no existe
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Dibujar el histograma de y
        plt.figure(figsize=(10, 6))
        plt.hist(combined_node_y, bins=50, color='g', alpha=0.7)
        plt.xlabel('y')
        plt.ylabel('Events')
        plt.title('y de todos los gráficos')
        
        # Guardar o mostrar el gráfico
        if save_dir:
            plt.savefig(f"{save_dir}/combined_node_y.png")
        else:
            plt.show()

def main():
    # Cargar el conjunto de datos PyTorch Geometric
    dataset = torch.load("./tools/training/vix_graph_ALL_layers_onlypt.pkl")
    
    # Crear una instancia del visualizador
    visualizer = GraphVisualizer(dataset)
    
    # Dibujar todas las propiedades de los nodos de todos los gráficos juntos
    visualizer.draw_combined_node_properties(save_dir="./node_properties")
        
    # Dibujar todas las propiedades de y de todos los gráficos juntos
    visualizer.draw_combined_node_y(save_dir="./node_y")

    # Dibujar todas las propiedades de los edges de todos los gráficos juntos
    visualizer.draw_combined_edge_properties(save_dir="./edge_properties")


if __name__ == "__main__":
    main()