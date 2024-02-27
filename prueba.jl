using Images, ImageFiltering, FileIO, ImageView, ImageMorphology, ColorTypes, ImageDraw, Colors

# Load the image
img = load("Frames/frames2/frame_34.jpg")

# Convert the image to HSV color space
img_hsv = HSV.(img)
display(img_hsv)


# Define the yellow color range in HSV
yellow_low = HSV(20/360, 100/255, 155/255)
yellow_high = HSV(62/360, 255/255, 255/255)
print("YEA2")
# Create a binary mask for pixels within the yellow range
mask = (img_hsv .>= yellow_low) .& (img_hsv .<=yellow_high)

print("YEA1")
# Convert mask to grayscale image for morphological operations
mask_gray = Gray.(mask)

# Perform morphological operations
kernel = Kernel.ellipse((2,2))
dilation_img = dilate(mask_gray, kernel)
erosion_img = erode(dilation_img, kernel)

circular_kernel = Kernel.ellipse((1,1))
erosion2_img = erode(erosion_img, circular_kernel)
dilation2_img = dilate(erosion2_img, circular_kernel)

# Save the modified image
imshow(dilation2_img)
