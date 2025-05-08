import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import psycopg2, osgeo.ogr, osgeo.gdal, osgeo.osr
import geopandas as gp
import shapely
from shapely.geometry import Polygon, LineString, Point
from utils import plot_polygons, get_class_definitions,grid_project


#Connect to database
conn=psycopg2.connect(host="postgis",database="gisdata",user="magsurveys",password="FinnGraemeChrys17")
cur=conn.cursor()

#select table
table = "interpretation_poly"
grid_size = 10 #m
image_size = 100 #m
image_radius = image_size/2 - 20
image_radius_wide = math.sqrt((image_size/2)**2 + (image_size/2)**2)
out_dir = r'/mnt/field/test/ml/cg'
class_definitions = get_class_definitions()

#find project IDs
#cur.execute("""SELECT DISTINCT project_id FROM %s """%(table))
#project_ids = cur.fetchall()
#
#for project_id in project_ids:
#    print(project_id)
#    plot_polygons(conn,project_id[0])

project_id = 'MSSP500'


cur.execute("""CREATE TABLE project AS SELECT * FROM %s WHERE project_id LIKE '%s'"""%(table,project_id))

polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project;",
    conn,
    geom_col='geom',
    coerce_float=True)

cur.execute("""DROP TABLE project""")

xv,yv = grid_project(polygons,grid_size = 10)

plot_polygons(conn, 'MSSP500', out_dir )
plot_polygons(conn, 'MSSP500', out_dir , image_name = 'grid', image_locations = [xv,yv], )

class_definitions = get_class_definitions()

data_points = []
for i,a in enumerate(xv):
    x = xv[i]
    y = yv[i]
    for j,b in enumerate(x):
        point = Point(x[j], y[j])
        d = polygons.distance(point)
        if sum(d<image_radius)>0:
            
            filtered_polygons = polygons[d<image_radius]
            filtered_polygons_wide = polygons[d<image_radius_wide]

            for c in class_definitions:
                Include = list(set(c['Include']) & set(filtered_polygons['interp_code']))
                Exclude = list(set(c['Exclude']) & set(filtered_polygons['interp_code']))

                if len(Include)>0 and len(Exclude)==0:
                    data_points.append([project_id,x[j],y[j],c['Description']])
output_data_points = pd.DataFrame(data_points,columns = ['Project Code', 'X','Y','Class'])

output_data_points.to_csv(os.path.join(out_dir,project_id+'_data_points.csv'),index=False)