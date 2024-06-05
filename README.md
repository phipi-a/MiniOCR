# MiniOCR
## install
```bash
pip install -e .
```
## usage 
```python

import MiniOCR

model = MiniOCR.Model()
model.predict('path/to/image.jpg')
```
output:
```python
[{'text': '6A', 'score': 0.8828625132332668},
 {'text': '6N', 'score': 0.8368727362876172},
 {'text': 'GA', 'score': 0.8302653136825455},
 {'text': '68', 'score': 0.8176551276281536},
 {'text': '6K', 'score': 0.8156228782897479},
 {'text': '6k', 'score': 0.8156228782897479},
 ...
]
```
