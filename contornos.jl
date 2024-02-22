using OpenCV
using Images
using ImageMorphology

imagen = OpenCV.imread("frames/frame23_83.png")

hsv = OpenCV.cvtColor(imagen, OpenCV.COLOR_BGR2HSV)

amarillo_bajo = OpenCV.Scalar(20, 100, 155)
amarillo_alto = OpenCV.Scalar(62, 255, 255)

mascara = OpenCV.inRange(hsv, amarillo_bajo, amarillo_alto)

kernel = OpenCV.getStructuringElement(OpenCV.MORPH_ELLIPSE, (2,2))
dilatacion = OpenCV.dilate(mascara, kernel, iterations=1)
erosion = OpenCV.erode(dilatacion, kernel, iterations=1)
circular_kernel = OpenCV.getStructuringElement(OpenCV.MORPH_ELLIPSE, (1,1))
erosion2 = OpenCV.erode(erosion, circular_kernel, iterations=1)
dilatacion2 = OpenCV.dilate(erosion2, circular_kernel, iterations=1)

contours, hierarchy = OpenCV.findContours(dilatacion, OpenCV.RETR_EXTERNAL, OpenCV.CHAIN_APPROX_SIMPLE)

min_area = 50
max_area = 2000
contours_filtrados = filter(cnt -> min_area < OpenCV.contourArea(cnt) < max_area, contours)

drawing = zeros(UInt8, (size(imagen, 1), size(imagen, 2), 3))
for cnt in contours_filtrados
    color = OpenCV.Scalar(255, 255, 255)
    OpenCV.drawContours(drawing, [cnt], -1, color, 2)
end

OpenCV.imshow("Original", imagen)
OpenCV.imshow("Mascara", mascara)

OpenCV.imshow("Erosion", erosion)

OpenCV.imshow("dilatacion2", dilatacion2)
img = Images.colorview(RGB, permutedims(dilatacion2, (3, 1, 2)))
save("frames_d/frame23_83_d.png", img)
OpenCV.waitKey(0)
OpenCV.destroyAllWindows()

