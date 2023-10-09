import numpy as np
proportions = np.random.dirichlet(
                        np.repeat(0.1, 10)
                    )
print(proportions)
