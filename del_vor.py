#MachNEase
#Author : Namitha Guruprasad
#LinkedIn : linkedin.com/in/namitha-guruprasad-216362155
#Import important libraries
import cv2
import numpy as np
import random

#to check if the point is inside the plane / region defined
def region(plane, point): # to check if the point is inside the plane / region defined
    if point[0] < plane[0]:
        return False
    elif point[1] < plane[1]:
        return False
    elif point[0] > plane[2]:
        return False
    elif point[1] > plane[3]:
        return False
    return True

#delaunay triangulation
def delaunay_triangulation(image, subdiv, color):
    #subdivision surface evaluation for the image (shape, surface etc)
    #color for drawing of lines for the traingulation in the region
    triangles = subdiv.getTriangleList();
    #triangle list is generated for the subdivisioned area
    size = image.shape
    #image size is calculated
    p = (0, 0, size[1], size[0])
    #plane demarkation to check if the points lie in the region
    for tr in triangles:
        p1 = (tr[0], tr[1])
        p2 = (tr[2], tr[3])
        p3 = (tr[4], tr[5])
        if region(p, p1) and region(p, p2) and region(p, p3):
            #if the points lie in the plane, then connect the dots
            cv2.line(image, p1, p2, color, 1, cv2.LINE_AA, 0)
            cv2.line(image, p2, p3, color, 1, cv2.LINE_AA, 0)
            cv2.line(image, p3, p1, color, 1, cv2.LINE_AA, 0)


#voronoi diagram
def draw_voronoi(image, subdiv):
    #tesellating the surface of the image with polygons for better subdivisions of the planes
    (facets, centers) = subdiv.getVoronoiFacetList([])
    #points in the plane - set of seeds
    #each seed - consists of region with all points closer to the seed than any other
    #similar to nearest neighbour algorithm
    array_colors = [[255,0,255],[255,248,220],[188,143,143],[0,255,255],[240,230,140]]
    #colors for the set of planes 
    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        color = array_colors[random.randint(0,4)]
        #random colors for different polygons 
        cv2.fillConvexPoly(image, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(image, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(image, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)


#obtain the image feed
img = cv2.imread("harvey.jpg");
#create a copy / duplicate of the image 
dup = img.copy();
#obtain image size
size = img.shape
#creating a plane
plane = (0, 0, size[1], size[0])
#creating object instance - for subdivisions of the regions 
subdiv = cv2.Subdiv2D(plane);

#extract array of points to check for its validity in the region / plane
points = []
#read the landmark coordinates from the "txt" file for validation
with open("harv.txt") as file :
    for line in file :
        x, y = line.split()
        points.append((int(x), int(y)))

for p in points :
    subdiv.insert(p)
    #animate the triangulation
    if True :
        copy = dup.copy()
        #construct delaunay triangles 
        delaunay_triangulation(copy, subdiv, (0, 255, 255));
        cv2.imshow('Delaunay Triangulation', copy)
        cv2.waitKey(100)

delaunay_triangulation( img, subdiv, (0, 255, 255) );

#marks the points on the image
for p in points:
    cv2.circle(img, p, 2, (255,0,0), cv2.FILLED, cv2.LINE_AA, 0)

#construct voronoi diagram
img_voronoi = np.zeros(img.shape, dtype = img.dtype)
draw_voronoi(img_voronoi,subdiv)

#display the windows
cv2.imshow('Delaunay Triangulation',img)
cv2.imshow('Voronoi Diagram',img_voronoi)
cv2.waitKey(0)





