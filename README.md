# Connect4

Proyecto de Fundamentos de la Inteligencia Artificial 2025-2

Juan Diego Lemus Rey - 0000243911
Carlos Andrés Zuluaga Mora - 0000272129

---

## **Cronograma preliminar**

### **Jueves 13 – Plan de acción**

- Definir el plan de trabajo.
- Alinear tareas, tiempos y responsabilidades.

### **Viernes 14 – GitHub y estructura**

- Organizar el repositorio.
- Dejar la estructura base lista y funcional.

### **Sábado 15 – Test del Gradescope**

- Revisar cómo está organizando el Gradescope.
- Validar entradas, salidas y formato de ejecución.

### **Domingo 16 – Base del agente**

- Construir toda la base del agente bien armada.

### **Lunes 17 – Métrica mínima**

- Pasar el Gradescope.

### **Martes 18 – Iteración del plan**

- Revisar cómo nos fue.
- Ajustar el plan, agregar lo que haga falta.
- Ver si ya se puede empezar a entrenar.
- Conseguir feedback de **Félix y/o Sergio**.

### **Miércoles 19 – Feedback**

- Conseguir nuevo feedback de **Félix y/o Sergio**.
- Ideal si se habla con ambos para comparar opiniones.

### **Miércoles (tarde) y Jueves 20 – Implementación**

- Implementar cualquier sugerencia recibida.
- Comenzar a entrenar offline hasta donde alcance.

### **Viernes 21 – Domingo 23 – Entrenamiento y métricas**

- Entrenar de forma continua.
- Ir monitoreando métricas de desempeño.

### **Domingo 23 (tarde, ojalá ~6 PM) – GitHub y entrega**

- Organizar todo lo del GitHub.
- Verificar rúbricas y detalles finales.
- Dejar todo preparado para la entrega.

### **Lunes 24 – Torneo**

- Ganarle a todos en el torneo.
- Celebrar con una hamburguesa.

### **Martes 25 – Sondeo**

- Preguntar a los del martes cómo les fue.
- Evaluar qué tan duro está calificando Félix.
- Decidir si presentar con Félix o con Sergio el miércoles.

### **Miércoles 26 – Presentación final**

- Presentar.

---

## Agenda

### **Jueves 13:**

- Entender pla plantilla
- Definir preguntas tutoría:

1. ¿Detalles del funcionamiento de los JSON? Explíqueme
2. ¿Qué quieren que hagamos con los groups?
3. Defina qué son los pesos, ¿Cuáles existen?
4. ¿Hay algún problema si da mos indicaciones al agente basados en estrategias exitosas fundamentales en el connect4?
5. Algun problema con usar bases de datos de partidas (no entrenamientos)
6. ¿Qué se sube al gradescopre?

### **Viernes 14:**

- Crear repo
- Subir los archivos iniciales
- Preguntar a Sergio
- Subir algo satisfactorio a gradescope
- Basados en supuestos crear una primera política

### **Sábado 15:**

- Intentos preliminares de un agente que sigue reglas:

  - Regla 1 si existe 3 en línea independientemente de qué jugador es, la toma
    - Si es propia, gana. Si es del rival, bloquea
  - Regla 2 busca hacer 3 en línea
  - Regla 3 busca la columna 3, estrategia real del juego
  - Regla 4 si todo lo demás falla, usa aleatorio

- Notas:
  - Gradescope colapsa si tiene la decoración y el import @override
  - Costó más de lo esperado la regla 1 para los 3 en línea que no son en vertical [falta aplicar]
  - Regla 2 no se implementó aún

### **Domingo 16:**

- Se tomó la decisión de dejar de iterar manualmente por el límite de tiempo así que se recurió a Inteligencia Artificial paraproseguir
- Github Copilot generó una solución E-Greedy que almacena Off-line Q-Tables en formato .pkl y comprimidos .pkl.gz

  - La solución fue entrenada hasta 50mil veces y logra resultados de hasta 90% jugando mil partidas en evaluate_agent.py

  - Se optimizaron los resultados usando las reglas descritas anteriormente, queda:

    - Regla 1, si hay 3 en línea vertical lo toma
    - Regla 2, juega la política con entrenamiento offline e-greedy buscando las tablas .pkl o .pkl.gz
    - Regla 3, si no tiene nada claro juega al centro
    - Regla 4, si el centro tampoco está disponible juega al azar

  - De esta forma logró 96% con sólo 5mil entrenamientos, así que se usaron 1 millón de entrenamientos para probar y se logró 100% en mil partidas con evaluate_agent.py

- Notas:
  - Almacenar en .pkl pesa demasiado para subir a gradescope;
  - Además parece que no tiene capacidad de usar ninguno de los formatos de entrenamiento en Q-Tables usados hasta ahora en Gradescope así que en cualquier caso, sigue jugando con las reglas o en aleatorio.
    **PREGUNTAR A SERGIO COMO FUNCIONA EXACTAMENTE EL ALMACENADO DE ESOS DATOS EN EL GRADESCOPE**

### **Lunes 17:**

- Nueva iteración teniendo en cuenta que no ha sido posible almacenar datos Off-line, se implementa un MCTS para tomar las decisiones online, de esa forma se evita el conflicto de archivos en Gradescope, para sacar el requisito de subir algo a la plataforma mientras se piensa alguna alternativa más dedicada.

- El MCTS tal como está logra un 100% sin entrenamiento pero tarda mucho en ejecutar.

- Se crearon copias de las políticas actuales, Random, RandomBeater, RLP, MCTS hasta 16 jugadores ya que constantemente aparecía error de Invalid Match: 2 BYEs match... Jugando con esta cantidad de jugadores siempre ganaba alguna copia de MCTS así que se decidió subir esa policy al Gradescope

- A pesar del tiempo, logra cumplir el Gradescope extendiendo el límite de tiempo.

### **Martes 18:**
- Actualizaron el Gradescope, el MCTS tarda demasiado en ejecutar, pero el RandomBeater original, sí pasa la prueba, nos quemados con ese mejor.

### **Domingo 23:**
- Se añade el documento que incluye la guía de uso y los datos necesarios para ejecuatar los archivos del torneo. Dicha información se encuentra en el documento de tal nombre, cuyo enlace es el siguiente: https://github.com/JuanDLemus/Connect4/blob/ee2a1161172a88199a09c9a05314a69fc4c7eaaa/Gu%C3%ADa%20de%20uso%20y%20datos%20necesarios%20para%20ejecutar.docx