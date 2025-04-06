'''
파이썬은 메모리 관리를 두 가지 방식으로 함.
- 참조 카운트
- 가비지 컬렉션

참조 카운트
- 모든 객체는 참조 카운트 0인 상태로 생성, 다른 객체가 참조할 때마다 1씩 증가함.
- 객체에 대한 참조가 끊길 때마다 1만큼 감소하며 0이 되면 파이썬 인터프리터가 회수함.

참조 카운트 증가
- 대입 연산자 사용할 때
- 함수에 인수로 전달할 때
- 컨테이너 타입 객체(리스트, 튜플, 클래스 등)에 추가할 때

순환 참조의 경우 참조 카운트 방식이 해결할 수 없는 경우 있음.
'''

import weakref
from memory_profiler import profile
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
    
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대 수 기록하는 변수
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()
        
        def add_func(f): # 
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)
    
    def cleargrad(self):
        self.grad = None
 
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in  inputs]) # 입력 변수가 둘 이상일 때 generation 큰 거 선택
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        # self.outputs = outputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
   
def add(x0, x1):
    return Add()(x0, x1)

@profile
def run_experiment():
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
        y.backward()
        
if __name__ == '__main__':
    run_experiment()