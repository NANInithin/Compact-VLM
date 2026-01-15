from PIL import Image
import numpy as np

def create_real_trap():
    print("🍌 Processing 'banana_real.jpg'...")
    
    try:
        img = Image.open("data/banana_real.jpg").convert("RGBA")
    except FileNotFoundError:
        print("❌ Error: Could not find 'data/banana_real.jpg'. Please download it first!")
        return

    # Convert to Numpy Array
    arr = np.array(img)
    
    # SEPARATE CHANNELS
    red, green, blue, alpha = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]

    # DEFINING THE MASK
    # We want pixels that are "Yellow-ish" but NOT "White" or "Black/Background"
    # Yellow = High Red, High Green, Low Blue
    # White  = High Red, High Green, High Blue
    
    # Logic: Red and Green are high (>100), but Blue is significantly lower than Green
    mask = (red > 100) & (green > 100) & (blue < green * 0.8)

    # APPLY COLOR SHIFT ONLY TO MASKED PIXELS
    # To make Yellow (R+G) into Purple (R+B):
    # 1. Keep Red high
    # 2. Drop Green (swap it with Blue)
    # 3. Boost Blue
    
    new_arr = arr.copy()
    
    # Swap Green and Blue channels for the banana pixels
    new_arr[mask, 1] = blue[mask]   # Green takes the low Blue values
    new_arr[mask, 2] = green[mask]  # Blue takes the high Green values

    # Save
    purple_banana = Image.fromarray(new_arr)
    purple_banana.save("data/banana_purple_real.png")
    print("✅ Created 'data/banana_purple_real.png' - Check if it looks like a purple fruit!")

if __name__ == "__main__":
    create_real_trap()