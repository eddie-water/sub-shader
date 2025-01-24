# Define some sample functions
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

def multiply(x, y=1):
    return x * y

class Foo():
    def __init__(self):
        pass

    def bar(self, number: int):
        print("foo_bar:", number)
        return number

foo = Foo()

# Create a list of functions with their arguments (as tuples)
function_list = [
    (greet,     ("Alice",)),        # Single argument requires a trailing comma in tuple
    (add,       (5, 3)),            # Two positional arguments
    (multiply,  (4,), {'y': 2}),    # Mixing positional and keyword arguments
    (foo.bar,   (10,))
]

# Loop through the list and call each function with its arguments
for item in function_list:
    func = item[0]
    args = item[1] if len(item) > 1 else ()
    kwargs = item[2] if len(item) > 2 else {}

    result = func(*args, **kwargs)
    print(f"{func.__name__}{args} {kwargs} -> {result}")
