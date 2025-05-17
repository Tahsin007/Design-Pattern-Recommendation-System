import base64
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

class PatternVisualizer:
    """Class for generating class diagrams for design patterns"""
    
    def __init__(self):
        # Dictionary storing the pattern structures
        self.pattern_structures = {
            "Singleton": self._singleton_structure,
            "Factory Method": self._factory_method_structure,
            "Abstract Factory": self._abstract_factory_structure,
            "Builder": self._builder_structure,
            "Prototype": self._prototype_structure,
            "Adapter": self._adapter_structure,
            "Bridge": self._bridge_structure,
            "Composite": self._composite_structure,
            "Decorator": self._decorator_structure,
            "Facade": self._facade_structure,
            "Flyweight": self._flyweight_structure,
            "Proxy": self._proxy_structure,
            "Chain of Responsibility": self._chain_of_responsibility_structure,
            "Command": self._command_structure,
            "Iterator": self._iterator_structure,
            "Mediator": self._mediator_structure,
            "Memento": self._memento_structure,
            "Observer": self._observer_structure,
            "State": self._state_structure,
            "Strategy": self._strategy_structure,
            "Template Method": self._template_method_structure,
            "Visitor": self._visitor_structure
        }
    
    def generate_diagram(self, pattern_name, problem_context=None):
        """Generate a class diagram for the given pattern
        
        Args:
            pattern_name: Name of the design pattern
            problem_context: Optional context from the user problem to customize diagram
            
        Returns:
            SVG string representation of the diagram
        """
        # Check if we have the pattern structure
        if pattern_name not in self.pattern_structures:
            return f"Pattern '{pattern_name}' not found in visualizer"
        
        # Get the pattern structure function
        structure_func = self.pattern_structures[pattern_name]
        
        # Generate the diagram using that function
        return structure_func(problem_context)
    
    def _draw_class_diagram(self, G, pos, class_details, relationships):
        """Create a UML-style class diagram
        
        Args:
            G: NetworkX graph
            pos: Node positions
            class_details: Dictionary mapping class name to {"attributes": [], "methods": []}
            relationships: List of tuples (source, target, relationship_type)
            
        Returns:
            SVG string of the diagram
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Drawing nodes (classes)
        for node in G.nodes():
            details = class_details.get(node, {"attributes": [], "methods": []})
            attributes = details.get("attributes", [])
            methods = details.get("methods", [])
            
            # Calculate box dimensions based on content
            width = max(len(node), 
                        max([len(attr) for attr in attributes] + [0]),
                        max([len(method) for method in methods] + [0])) * 0.13 + 0.5
            height = 0.8 + len(attributes) * 0.2 + len(methods) * 0.2
            
            # Draw class box
            rect = plt.Rectangle((pos[node][0] - width/2, pos[node][1] - height/2), 
                                width, height, fill=True, color='lightblue', 
                                alpha=0.8, ec='black', zorder=1)
            ax.add_patch(rect)
            
            # Draw class name
            ax.text(pos[node][0], pos[node][1] + height/2 - 0.3, node, 
                   fontsize=12, ha='center', fontweight='bold')
            
            # Draw separator line after class name
            line_y = pos[node][1] + height/2 - 0.4
            plt.plot([pos[node][0] - width/2, pos[node][0] + width/2], 
                     [line_y, line_y], 'k-', lw=1)
            
            # Draw attributes
            for i, attr in enumerate(attributes):
                ax.text(pos[node][0] - width/2 + 0.1, line_y - 0.2 - i * 0.2, 
                       attr, fontsize=10, ha='left')
            
            # Draw separator line before methods
            if methods:
                method_line_y = line_y - 0.2 - len(attributes) * 0.2
                plt.plot([pos[node][0] - width/2, pos[node][0] + width/2], 
                         [method_line_y, method_line_y], 'k-', lw=1)
                
                # Draw methods
                for i, method in enumerate(methods):
                    ax.text(pos[node][0] - width/2 + 0.1, method_line_y - 0.2 - i * 0.2, 
                           method, fontsize=10, ha='left')
        
        # Drawing edges (relationships)
        edge_styles = {
            "inheritance": {"style": "solid", "arrow": "normal", "arrowstyle": "-|>"},
            "implementation": {"style": "dashed", "arrow": "normal", "arrowstyle": "-|>"},
            "association": {"style": "solid", "arrow": "normal", "arrowstyle": "->"},
            "dependency": {"style": "dashed", "arrow": "normal", "arrowstyle": "->"},
            "aggregation": {"style": "solid", "arrow": "diamond", "arrowstyle": "-o"},
            "composition": {"style": "solid", "arrow": "diamond", "arrowstyle": "-*"}
        }
        
        # Draw relationships
        for source, target, rel_type in relationships:
            if source not in pos or target not in pos:
                continue
                
            style = edge_styles.get(rel_type, edge_styles["association"])
            linestyle = "dashed" if style["style"] == "dashed" else "solid"
            
            arrow = FancyArrowPatch(
                pos[source], pos[target],
                connectionstyle="arc3,rad=0.1",
                arrowstyle=style["arrowstyle"],
                mutation_scale=20,
                lw=1,
                linestyle=linestyle
            )
            
            # Add edge labels for clarity
            mid_x = (pos[source][0] + pos[target][0]) / 2
            mid_y = (pos[source][1] + pos[target][1]) / 2
            if rel_type != "association":  # Skip labels for basic associations
                ax.text(mid_x, mid_y, rel_type, fontsize=8, 
                       ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
            
            ax.add_patch(arrow)
            
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        
        # Convert to SVG string
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        buf.seek(0)
        svg_string = buf.getvalue().decode()
        plt.close(fig)
        
        return svg_string
    
    def _singleton_structure(self, context=None):
        """Generate Singleton pattern diagram"""
        G = nx.DiGraph()
        G.add_node("Client")
        G.add_node("Singleton")
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Singleton": (3, 0)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": ["main()"]
            },
            "Singleton": {
                "attributes": ["- static instance: Singleton",
                              "- data: type"],
                "methods": ["- Singleton()",
                           "+ static getInstance(): Singleton",
                           "+ operation()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Singleton", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _factory_method_structure(self, context=None):
        """Generate Factory Method pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Creator", "ConcreteCreator", "Product", "ConcreteProduct"])
        
        # Define positions
        pos = {
            "Client": (0, 2),
            "Creator": (3, 2),
            "ConcreteCreator": (3, 0),
            "Product": (6, 2),
            "ConcreteProduct": (6, 0)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": ["operation()"]
            },
            "Creator": {
                "attributes": [],
                "methods": ["+ factoryMethod(): Product",
                           "+ operation()"]
            },
            "ConcreteCreator": {
                "attributes": [],
                "methods": ["+ factoryMethod(): Product"]
            },
            "Product": {
                "attributes": [],
                "methods": ["+ operation()"]
            },
            "ConcreteProduct": {
                "attributes": [],
                "methods": ["+ operation()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Creator", "association"),
            ("ConcreteCreator", "Creator", "inheritance"),
            ("ConcreteProduct", "Product", "inheritance"),
            ("Creator", "Product", "dependency"),
            ("ConcreteCreator", "ConcreteProduct", "dependency")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _builder_structure(self, context=None):
        """Generate Builder pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Director", "Builder", "ConcreteBuilder", "Product"])
        
        # Define positions
        pos = {
            "Director": (0, 1),
            "Builder": (3, 1),
            "ConcreteBuilder": (3, -1),
            "Product": (6, -1)
        }
        
        # Define class details
        class_details = {
            "Director": {
                "attributes": ["- builder: Builder"],
                "methods": ["+ construct()",
                           "+ setBuilder(Builder)"]
            },
            "Builder": {
                "attributes": [],
                "methods": ["+ buildPartA()",
                           "+ buildPartB()",
                           "+ getResult()"]
            },
            "ConcreteBuilder": {
                "attributes": ["- product: Product"],
                "methods": ["+ buildPartA()",
                           "+ buildPartB()",
                           "+ getResult(): Product"]
            },
            "Product": {
                "attributes": ["- parts: List"],
                "methods": []
            }
        }
        
        # Define relationships
        relationships = [
            ("Director", "Builder", "association"),
            ("ConcreteBuilder", "Builder", "inheritance"),
            ("ConcreteBuilder", "Product", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _adapter_structure(self, context=None):
        """Generate Adapter pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Target", "Adapter", "Adaptee"])
        
        # Define positions
        pos = {
            "Client": (0, 1),
            "Target": (3, 1),
            "Adapter": (3, -1),
            "Adaptee": (6, -1)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": ["+ request()"]
            },
            "Target": {
                "attributes": [],
                "methods": ["+ request()"]
            },
            "Adapter": {
                "attributes": ["- adaptee: Adaptee"],
                "methods": ["+ request()"]
            },
            "Adaptee": {
                "attributes": [],
                "methods": ["+ specificRequest()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Target", "association"),
            ("Adapter", "Target", "inheritance"),
            ("Adapter", "Adaptee", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _bridge_structure(self, context=None):
        """Generate Bridge pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Abstraction", "RefinedAbstraction", "Implementor", "ConcreteImplementorA", "ConcreteImplementorB"])
        
        # Define positions
        pos = {
            "Abstraction": (1, 2),
            "RefinedAbstraction": (1, 0),
            "Implementor": (5, 2),
            "ConcreteImplementorA": (4, 0),
            "ConcreteImplementorB": (6, 0)
        }
        
        # Define class details
        class_details = {
            "Abstraction": {
                "attributes": ["- implementor: Implementor"],
                "methods": ["+ operation()"]
            },
            "RefinedAbstraction": {
                "attributes": [],
                "methods": ["+ operation()"]
            },
            "Implementor": {
                "attributes": [],
                "methods": ["+ operationImpl()"]
            },
            "ConcreteImplementorA": {
                "attributes": [],
                "methods": ["+ operationImpl()"]
            },
            "ConcreteImplementorB": {
                "attributes": [],
                "methods": ["+ operationImpl()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("RefinedAbstraction", "Abstraction", "inheritance"),
            ("Abstraction", "Implementor", "association"),
            ("ConcreteImplementorA", "Implementor", "inheritance"),
            ("ConcreteImplementorB", "Implementor", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _composite_structure(self, context=None):
        """Generate Composite pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Component", "Leaf", "Composite"])
        
        # Define positions
        pos = {
            "Client": (0, 1),
            "Component": (3, 1),
            "Leaf": (1.5, -1),
            "Composite": (4.5, -1)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "Component": {
                "attributes": [],
                "methods": ["+ operation()",
                           "+ add(Component)",
                           "+ remove(Component)",
                           "+ getChild(int)"]
            },
            "Leaf": {
                "attributes": [],
                "methods": ["+ operation()"]
            },
            "Composite": {
                "attributes": ["- children: List<Component>"],
                "methods": ["+ operation()",
                           "+ add(Component)",
                           "+ remove(Component)",
                           "+ getChild(int)"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Component", "association"),
            ("Leaf", "Component", "inheritance"),
            ("Composite", "Component", "inheritance"),
            ("Composite", "Component", "composition")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _decorator_structure(self, context=None):
        """Generate Decorator pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Component", "ConcreteComponent", "Decorator", "ConcreteDecoratorA", "ConcreteDecoratorB"])
        
        # Define positions
        pos = {
            "Component": (3, 2),
            "ConcreteComponent": (1, 0),
            "Decorator": (5, 0),
            "ConcreteDecoratorA": (4, -2),
            "ConcreteDecoratorB": (6, -2)
        }
        
        # Define class details
        class_details = {
            "Component": {
                "attributes": [],
                "methods": ["+ operation()"]
            },
            "ConcreteComponent": {
                "attributes": [],
                "methods": ["+ operation()"]
            },
            "Decorator": {
                "attributes": ["- component: Component"],
                "methods": ["+ operation()"]
            },
            "ConcreteDecoratorA": {
                "attributes": ["- additionalState"],
                "methods": ["+ operation()"]
            },
            "ConcreteDecoratorB": {
                "attributes": [],
                "methods": ["+ operation()",
                           "+ addedBehavior()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("ConcreteComponent", "Component", "inheritance"),
            ("Decorator", "Component", "inheritance"),
            ("Decorator", "Component", "association"),
            ("ConcreteDecoratorA", "Decorator", "inheritance"),
            ("ConcreteDecoratorB", "Decorator", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _facade_structure(self, context=None):
        """Generate Facade pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Facade", "SubsystemA", "SubsystemB", "SubsystemC"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Facade": (3, 0),
            "SubsystemA": (6, 1),
            "SubsystemB": (6, 0),
            "SubsystemC": (6, -1)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "Facade": {
                "attributes": ["- subsystemA: SubsystemA",
                              "- subsystemB: SubsystemB",
                              "- subsystemC: SubsystemC"],
                "methods": ["+ operation()"]
            },
            "SubsystemA": {
                "attributes": [],
                "methods": ["+ operationA()"]
            },
            "SubsystemB": {
                "attributes": [],
                "methods": ["+ operationB()"]
            },
            "SubsystemC": {
                "attributes": [],
                "methods": ["+ operationC()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Facade", "association"),
            ("Facade", "SubsystemA", "association"),
            ("Facade", "SubsystemB", "association"),
            ("Facade", "SubsystemC", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _flyweight_structure(self, context=None):
        """Generate Flyweight pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "FlyweightFactory", "Flyweight", "ConcreteFlyweight", "UnsharedConcreteFlyweight"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "FlyweightFactory": (3, 0),
            "Flyweight": (6, 0),
            "ConcreteFlyweight": (5, -2),
            "UnsharedConcreteFlyweight": (7, -2)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "FlyweightFactory": {
                "attributes": ["- flyweights: Map"],
                "methods": ["+ getFlyweight(key)"]
            },
            "Flyweight": {
                "attributes": [],
                "methods": ["+ operation(extrinsicState)"]
            },
            "ConcreteFlyweight": {
                "attributes": ["- intrinsicState"],
                "methods": ["+ operation(extrinsicState)"]
            },
            "UnsharedConcreteFlyweight": {
                "attributes": ["- allState"],
                "methods": ["+ operation(extrinsicState)"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "FlyweightFactory", "association"),
            ("Client", "Flyweight", "association"),
            ("FlyweightFactory", "ConcreteFlyweight", "association"),
            ("ConcreteFlyweight", "Flyweight", "inheritance"),
            ("UnsharedConcreteFlyweight", "Flyweight", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _proxy_structure(self, context=None):
        """Generate Proxy pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Subject", "RealSubject", "Proxy"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Subject": (3, 0),
            "RealSubject": (2, -2),
            "Proxy": (4, -2)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "Subject": {
                "attributes": [],
                "methods": ["+ request()"]
            },
            "RealSubject": {
                "attributes": [],
                "methods": ["+ request()"]
            },
            "Proxy": {
                "attributes": ["- realSubject: RealSubject"],
                "methods": ["+ request()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Subject", "association"),
            ("RealSubject", "Subject", "inheritance"),
            ("Proxy", "Subject", "inheritance"),
            ("Proxy", "RealSubject", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _chain_of_responsibility_structure(self, context=None):
        """Generate Chain of Responsibility pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Handler", "ConcreteHandler1", "ConcreteHandler2"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Handler": (3, 0),
            "ConcreteHandler1": (2, -2),
            "ConcreteHandler2": (4, -2)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "Handler": {
                "attributes": ["- successor: Handler"],
                "methods": ["+ handleRequest()",
                           "+ setSuccessor(Handler)"]
            },
            "ConcreteHandler1": {
                "attributes": [],
                "methods": ["+ handleRequest()"]
            },
            "ConcreteHandler2": {
                "attributes": [],
                "methods": ["+ handleRequest()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Handler", "association"),
            ("Handler", "Handler", "association"),
            ("ConcreteHandler1", "Handler", "inheritance"),
            ("ConcreteHandler2", "Handler", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _command_structure(self, context=None):
        """Generate Command pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Invoker", "Command", "ConcreteCommand", "Receiver"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Invoker": (3, 1),
            "Command": (3, -1),
            "ConcreteCommand": (3, -3),
            "Receiver": (6, -3)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "Invoker": {
                "attributes": ["- command: Command"],
                "methods": ["+ setCommand(Command)",
                           "+ executeCommand()"]
            },
            "Command": {
                "attributes": [],
                "methods": ["+ execute()"]
            },
            "ConcreteCommand": {
                "attributes": ["- receiver: Receiver",
                              "- state"],
                "methods": ["+ execute()"]
            },
            "Receiver": {
                "attributes": [],
                "methods": ["+ action()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "ConcreteCommand", "association"),
            ("Invoker", "Command", "association"),
            ("ConcreteCommand", "Command", "inheritance"),
            ("ConcreteCommand", "Receiver", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _iterator_structure(self, context=None):
        """Generate Iterator pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Iterator", "ConcreteIterator", "Aggregate", "ConcreteAggregate"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Iterator": (3, 1),
            "ConcreteIterator": (3, -1),
            "Aggregate": (6, 1),
            "ConcreteAggregate": (6, -1)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": []
            },
            "Iterator": {
                "attributes": [],
                "methods": ["+ first()",
                           "+ next()",
                           "+ isDone()",
                           "+ currentItem()"]
            },
            "ConcreteIterator": {
                "attributes": [],
                "methods": ["+ first()",
                           "+ next()",
                           "+ isDone()",
                           "+ currentItem()"]
            },
            "Aggregate": {
                "attributes": [],
                "methods": ["+ createIterator()"]
            },
            "ConcreteAggregate": {
                "attributes": [],
                "methods": ["+ createIterator()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Iterator", "association"),
            ("Client", "Aggregate", "association"),
            ("ConcreteIterator", "Iterator", "inheritance"),
            ("ConcreteAggregate", "Aggregate", "inheritance"),
            ("ConcreteIterator", "ConcreteAggregate", "association"),
            ("ConcreteAggregate", "ConcreteIterator", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _mediator_structure(self, context=None):
        """Generate Mediator pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Mediator", "ConcreteMediator", "Colleague", "ConcreteColleague1", "ConcreteColleague2"])
        
        # Define positions
        pos = {
            "Mediator": (3, 2),
            "ConcreteMediator": (3, 0),
            "Colleague": (0, 0),
            "ConcreteColleague1": (-2, -2),
            "ConcreteColleague2": (2, -2)
        }
        
        # Define class details
        class_details = {
            "Mediator": {
                "attributes": [],
                "methods": ["+ notify(sender, event)"]
            },
            "ConcreteMediator": {
                "attributes": ["- colleague1: ConcreteColleague1",
                              "- colleague2: ConcreteColleague2"],
                "methods": ["+ notify(sender, event)"]
            },
            "Colleague": {
                "attributes": ["# mediator: Mediator"],
                "methods": ["+ setMediator(Mediator)"]
            },
            "ConcreteColleague1": {
                "attributes": [],
                "methods": ["+ send()",
                           "+ receive()"]
            },
            "ConcreteColleague2": {
                "attributes": [],
                "methods": ["+ send()",
                           "+ receive()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("ConcreteMediator", "Mediator", "inheritance"),
            ("ConcreteColleague1", "Colleague", "inheritance"),
            ("ConcreteColleague2", "Colleague", "inheritance"),
            ("Colleague", "Mediator", "association"),
            ("ConcreteMediator", "ConcreteColleague1", "association"),
            ("ConcreteMediator", "ConcreteColleague2", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _memento_structure(self, context=None):
        """Generate Memento pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Originator", "Memento", "Caretaker"])
        
        # Define positions
        pos = {
            "Originator": (0, 0),
            "Memento": (3, 0),
            "Caretaker": (6, 0)
        }
        
        # Define class details
        class_details = {
            "Originator": {
                "attributes": ["- state"],
                "methods": ["+ setMemento(Memento)",
                           "+ createMemento(): Memento"]
            },
            "Memento": {
                "attributes": ["- state"],
                "methods": ["+ getState()",
                           "+ setState(state)"]
            },
            "Caretaker": {
                "attributes": ["- memento: Memento"],
                "methods": []
            }
        }
        
        # Define relationships
        relationships = [
            ("Originator", "Memento", "association"),
            ("Caretaker", "Memento", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _observer_structure(self, context=None):
        """Generate Observer pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Subject", "ConcreteSubject", "Observer", "ConcreteObserver"])
        
        # Define positions
        pos = {
            "Subject": (1, 1),
            "ConcreteSubject": (1, -1),
            "Observer": (5, 1),
            "ConcreteObserver": (5, -1)
        }
        
        # Define class details
        class_details = {
            "Subject": {
                "attributes": ["- observers: List<Observer>"],
                "methods": ["+ attach(Observer)",
                           "+ detach(Observer)",
                           "+ notify()"]
            },
            "ConcreteSubject": {
                "attributes": ["- subjectState"],
                "methods": ["+ getState()",
                           "+ setState()"]
            },
            "Observer": {
                "attributes": [],
                "methods": ["+ update()"]
            },
            "ConcreteObserver": {
                "attributes": ["- observerState",
                              "- subject: ConcreteSubject"],
                "methods": ["+ update()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("ConcreteSubject", "Subject", "inheritance"),
            ("ConcreteObserver", "Observer", "inheritance"),
            ("Subject", "Observer", "association"),
            ("ConcreteObserver", "ConcreteSubject", "association")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _state_structure(self, context=None):
        """Generate State pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Context", "State", "ConcreteStateA", "ConcreteStateB"])
        
        # Define positions
        pos = {
            "Context": (0, 0),
            "State": (3, 0),
            "ConcreteStateA": (2, -2),
            "ConcreteStateB": (4, -2)
        }
        
        # Define class details
        class_details = {
            "Context": {
                "attributes": ["- state: State"],
                "methods": ["+ request()",
                           "+ setState(State)"]
            },
            "State": {
                "attributes": [],
                "methods": ["+ handle()"]
            },
            "ConcreteStateA": {
                "attributes": [],
                "methods": ["+ handle()"]
            },
            "ConcreteStateB": {
                "attributes": [],
                "methods": ["+ handle()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Context", "State", "association"),
            ("ConcreteStateA", "State", "inheritance"),
            ("ConcreteStateB", "State", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _strategy_structure(self, context=None):
        """Generate Strategy pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Context", "Strategy", "ConcreteStrategyA", "ConcreteStrategyB"])
        
        # Define positions
        pos = {
            "Context": (0, 0),
            "Strategy": (3, 0),
            "ConcreteStrategyA": (2, -2),
            "ConcreteStrategyB": (4, -2)
        }
        
        # Define class details
        class_details = {
            "Context": {
                "attributes": ["- strategy: Strategy"],
                "methods": ["+ contextInterface()",
                           "+ setStrategy(Strategy)"]
            },
            "Strategy": {
                "attributes": [],
                "methods": ["+ algorithmInterface()"]
            },
            "ConcreteStrategyA": {
                "attributes": [],
                "methods": ["+ algorithmInterface()"]
            },
            "ConcreteStrategyB": {
                "attributes": [],
                "methods": ["+ algorithmInterface()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Context", "Strategy", "association"),
            ("ConcreteStrategyA", "Strategy", "inheritance"),
            ("ConcreteStrategyB", "Strategy", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _template_method_structure(self, context=None):
        """Generate Template Method pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["AbstractClass", "ConcreteClass"])
        
        # Define positions
        pos = {
            "AbstractClass": (2, 1),
            "ConcreteClass": (2, -1)
        }
        
        # Define class details
        class_details = {
            "AbstractClass": {
                "attributes": [],
                "methods": ["+ templateMethod()",
                           "# primitiveOperation1()",
                           "# primitiveOperation2()"]
            },
            "ConcreteClass": {
                "attributes": [],
                "methods": ["# primitiveOperation1()",
                           "# primitiveOperation2()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("ConcreteClass", "AbstractClass", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _visitor_structure(self, context=None):
        """Generate Visitor pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Visitor", "ConcreteVisitor", "Element", "ConcreteElementA", "ConcreteElementB"])
        
        # Define positions
        pos = {
            "Visitor": (1, 2),
            "ConcreteVisitor": (1, 0),
            "Element": (5, 2),
            "ConcreteElementA": (4, 0),
            "ConcreteElementB": (6, 0)
        }
        
        # Define class details
        class_details = {
            "Visitor": {
                "attributes": [],
                "methods": ["+ visitElementA(ElementA)",
                           "+ visitElementB(ElementB)"]
            },
            "ConcreteVisitor": {
                "attributes": [],
                "methods": ["+ visitElementA(ElementA)",
                           "+ visitElementB(ElementB)"]
            },
            "Element": {
                "attributes": [],
                "methods": ["+ accept(Visitor)"]
            },
            "ConcreteElementA": {
                "attributes": [],
                "methods": ["+ accept(Visitor)",
                           "+ operationA()"]
            },
            "ConcreteElementB": {
                "attributes": [],
                "methods": ["+ accept(Visitor)",
                           "+ operationB()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("ConcreteVisitor", "Visitor", "inheritance"),
            ("ConcreteElementA", "Element", "inheritance"),
            ("ConcreteElementB", "Element", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _abstract_factory_structure(self, context=None):
        """Generate Abstract Factory pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["AbstractFactory", "ConcreteFactory1", "ConcreteFactory2", 
                         "AbstractProductA", "AbstractProductB", 
                         "ProductA1", "ProductA2", "ProductB1", "ProductB2"])
        
        # Define positions
        pos = {
            "AbstractFactory": (3, 3),
            "ConcreteFactory1": (1, 1),
            "ConcreteFactory2": (5, 1),
            "AbstractProductA": (1, -1),
            "AbstractProductB": (5, -1),
            "ProductA1": (0, -3),
            "ProductA2": (2, -3),
            "ProductB1": (4, -3),
            "ProductB2": (6, -3)
        }
        
        # Define class details
        class_details = {
            "AbstractFactory": {
                "attributes": [],
                "methods": ["+ createProductA()",
                           "+ createProductB()"]
            },
            "ConcreteFactory1": {
                "attributes": [],
                "methods": ["+ createProductA()",
                           "+ createProductB()"]
            },
            "ConcreteFactory2": {
                "attributes": [],
                "methods": ["+ createProductA()",
                           "+ createProductB()"]
            },
            "AbstractProductA": {
                "attributes": [],
                "methods": []
            },
            "AbstractProductB": {
                "attributes": [],
                "methods": []
            },
            "ProductA1": {
                "attributes": [],
                "methods": []
            },
            "ProductA2": {
                "attributes": [],
                "methods": []
            },
            "ProductB1": {
                "attributes": [],
                "methods": []
            },
            "ProductB2": {
                "attributes": [],
                "methods": []
            }
        }
        
        # Define relationships
        relationships = [
            ("ConcreteFactory1", "AbstractFactory", "inheritance"),
            ("ConcreteFactory2", "AbstractFactory", "inheritance"),
            ("ProductA1", "AbstractProductA", "inheritance"),
            ("ProductA2", "AbstractProductA", "inheritance"),
            ("ProductB1", "AbstractProductB", "inheritance"),
            ("ProductB2", "AbstractProductB", "inheritance"),
            ("ConcreteFactory1", "ProductA1", "dependency"),
            ("ConcreteFactory1", "ProductB1", "dependency"),
            ("ConcreteFactory2", "ProductA2", "dependency"),
            ("ConcreteFactory2", "ProductB2", "dependency")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)
    
    def _prototype_structure(self, context=None):
        """Generate Prototype pattern diagram"""
        G = nx.DiGraph()
        G.add_nodes_from(["Client", "Prototype", "ConcretePrototype1", "ConcretePrototype2"])
        
        # Define positions
        pos = {
            "Client": (0, 0),
            "Prototype": (3, 0),
            "ConcretePrototype1": (2, -2),
            "ConcretePrototype2": (4, -2)
        }
        
        # Define class details
        class_details = {
            "Client": {
                "attributes": [],
                "methods": ["+ operation()"]
            },
            "Prototype": {
                "attributes": [],
                "methods": ["+ clone()"]
            },
            "ConcretePrototype1": {
                "attributes": [],
                "methods": ["+ clone()"]
            },
            "ConcretePrototype2": {
                "attributes": [],
                "methods": ["+ clone()"]
            }
        }
        
        # Define relationships
        relationships = [
            ("Client", "Prototype", "association"),
            ("ConcretePrototype1", "Prototype", "inheritance"),
            ("ConcretePrototype2", "Prototype", "inheritance")
        ]
        
        return self._draw_class_diagram(G, pos, class_details, relationships)