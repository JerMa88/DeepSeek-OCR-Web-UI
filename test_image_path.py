#!/usr/bin/env python3

def test_function(test_image='./images/rec_sys.png'):
    print(f"Function received image path: '{test_image}'")
    print(f"Length: {len(test_image)}")
    print(f"Type: {type(test_image)}")
    
    import os
    print(f"File exists: {os.path.exists(test_image)}")
    return test_image

if __name__ == "__main__":
    print("Testing image path...")
    result = test_function()
    print(f"Result: '{result}'")