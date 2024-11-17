from osgeo import osr
import ReadWrite_h5

class GEOchange(object):

    def __init__(self, toEPSG):
        self.EPSG = toEPSG
        self.to_crs = osr.SpatialReference()
        self.to_crs.ImportFromEPSG(toEPSG)

    def run(self, dataset, im_proj, im_geotrans):
        self.im_width, self.im_height = ReadWrite_h5.get_RasterXY(dataset)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(im_proj.ExportToWkt())
        self.Transformation = osr.CoordinateTransformation(srs, self.to_crs)
        geotrans = self.setGeotrans(im_geotrans)
        return self.to_crs.ExportToWkt(), geotrans

    def setGeotrans(self, im_geotrans):
        lon, lat = self.imagexy2geo(im_geotrans, 0, 0)
        coords00 = self.Transformation.TransformPoint(lat, lon)
        lon, lat = self.imagexy2geo(im_geotrans, self.im_height, 0)
        coords01 = self.Transformation.TransformPoint(lat, lon)
        lon, lat = self.imagexy2geo(im_geotrans, 0, self.im_width)
        coords10 = self.Transformation.TransformPoint(lat, lon)
        print('coord00', coords00)
        print('coord01', coords01)
        print('coord10', coords10)
        print('width, height', self.im_width, self.im_height)
        trans = [0 for i in range(6)]
        trans[0] = coords00[0]
        trans[3] = coords00[1]
        trans[2] = (coords01[0] - trans[0]) / self.im_height
        trans[5] = (coords01[1] - trans[3]) / self.im_height
        trans[1] = (coords10[0] - trans[0]) / self.im_width
        trans[4] = (coords10[1] - trans[3]) / self.im_width
        return trans

    def imagexy2geo(self, im_geotrans, row, col):
        px = im_geotrans[0] + col * im_geotrans[1] + row * im_geotrans[2]
        py = im_geotrans[3] + col * im_geotrans[4] + row * im_geotrans[5]
        print('左上角经纬度', px, py)
        return px, py


if __name__ == '__main__':
    change = GEOchange(32650)



