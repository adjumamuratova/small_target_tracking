# Small_target_tracking

Задачей являлось отслеживание малоразмерных объектов (< 10 пикселей). В качестве детекторов точечных объектов использовались:
ORB-детектор, good features to Track, Байесовский классификатор. Трекинг осуществлялся с помощью венгерского алгоритма и 
калмановского фильтра или ECO-трекера.

## Запуск трекера:

###Для теста ORB-детектора:
  - check ORB,  Clusterization for ORB, Hungarian-ORB;
  - наметить мышью отслеживаемую цель;
  - горячая клавиша Play/Pause - Enter;
###Для теста ECO трекера:
  - check ECO;
  - накинуть roi мышью на отслеживаемую цель;
  - горячая клавиша Play/Pause - Enter;
