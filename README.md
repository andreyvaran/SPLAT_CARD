# SPLAT PHOTO ANALIZE

### Установка и сборка проекта 
1) ```cd backend``` 


#### Предполагается, что у пользователя уже установлен uv. Без него собрать пипец как тяжко. 


```uv python install 3.11.11```

```uv venv ```

``` source .venv/bin/activate```

#### Дальше фиксим прикол "ModuleNotFoundError: No module named 'mmcv._ext'"


```
uv pip uninstall torch torchvision numpy openmim mmengine mmcv mmdet
uv pip install torch==2.0.0
uv pip install torchvision==0.16.2
uv pip install numpy==1.26.4
uv pip install openmim

mim install mmengine
mim install mmdet

mim install "mmcv==2.0.0rc4"
```
Последняя штука может собираться минут 5-10. Фиг его знает почему. 

### Запуск
```docker compose up -d  postgres redis ```

```python run.py``` 

or 
```nohuo python run.py```
