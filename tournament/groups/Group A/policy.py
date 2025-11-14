import numpy as np
from connect4.policy import Policy
from typing import override


class RandomBeater(Policy):

    @override
    def mount(self) -> None:
        # No monta nada
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        
        # Regla 1:
        ## Si ya hay 3, gana o bloquea
        ### next_is_final encuentra si hay 3 en linea y tomamos el int como Bool para elegir la fila a jugar
        ya_hay_3_en_linea = self.next_is_final(self,s)
        if ya_hay_3_en_linea:
            # elige lacolumna para ganar o bloquear
            pass

        # Regla 2:
        ## Busque 3 en linea
        

        # Regla 3:
        ## Busca el centro



        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))
    
    def next_is_final(self):
        # Reciclado de connect_state con 3 en vez de 4
        for r in range(self.ROWS):
            for c in range(self.COLS):
                player = self.board[r, c]
                if player == 0:
                    continue

                # 3 en vertical, adicionalmente retorna la columna, valida:
                ## is_applicable
                if r + 3 < self.ROWS and all(self.board[r + i, c] == player for i in range(3)):
                    return (player, (c, None))
                
                # 3 en horizontal, adicionalmente retorna la columna anterior y siguiente, valida:
                ## columna anterior:
                #### estado actual sea 1 fila abajo del 3 en linea (al seleccionar queda en la misma fila del 3 en linea)
                #### no salga de los límites
                #### is_applicable
                ## columna siguiente:
                #### estado actual sea 1 fila abajo del 3 en linea (al seleccionar queda en la misma fila del 3 en linea)
                #### no salga de los límites
                #### is_applicable
                if c + 3 < self.COLS and all(self.board[r, c + i] == player for i in range(3)):
                    return (player, (cA, cS))
                
                # 3 en diagonal right-down, adicionalmente retorna la columna anterior y siguiente, valida:
                ## columna anterior:
                #### estado actual sea la misma fila del 3 en linea (al seleccionar queda en la fila justo arriba del 3 en linea)
                #### no salga de los límites
                #### is_applicable
                ## columna siguiente:
                #### estado actual sea 2 filas abajo del 3 en linea (al seleccionar queda en la fila justo abajo del 3 en linea)
                #### no salga de los límites
                #### is_applicable
                if (r + 3 < self.ROWS and c + 3 < self.COLS and all(self.board[r + i, c + i] == player for i in range(3))):
                    return (player, (cA, cS))
                
                # 3 en diagonal left-down, adicionalmente retorna la columna anterior y siguiente, valida:
                ## columna anterior:
                #### estado actual sea 2 filas abajo del 3 en linea (al seleccionar queda en la fila justo abajo del 3 en linea)
                #### no salga de los límites
                #### is_applicable
                ## columna siguiente:
                #### estado actual sea la misma fila del 3 en linea (al seleccionar queda en la fila justo arriba del 3 en linea)
                #### no salga de los límites
                #### is_applicable
                if (r + 3 < self.ROWS and c - 3 >= 0 and all(self.board[r + i, c - i] == player for i in range(3)) ):
                    return (player, (cA, cS))
        return (0, (None, None))