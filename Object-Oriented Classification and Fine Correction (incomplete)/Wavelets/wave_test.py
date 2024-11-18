# --coding:utf-8--
from osgeo import gdal
import numpy as np
from pywt import wavedec, waverec

def wavelet_decomp(image, wavelet):
  """
  对图像进行小波分解

  Args:
    image: 输入图像
    wavelet: 小波函数

  Returns:
    coefficients: 分解后的系数数组列表
  """

  # 计算小波系数
  coefficients = wavedec(image, wavelet)

  # 返回分解后的系数数组列表
  return coefficients


def wavelet_recon(coefficients, wavelet):
  """
  将分解后的子图像重构为原图像

  Args:
    coefficients: 分解后的系数数组列表
    wavelet: 小波函数

  Returns:
    reconst_image: 重构后的图像
  """

  # 合并分解后的系数数组
  coefficients = np.concatenate(coefficients, axis=1)

  # 使用小波逆变换重构图像
  reconst_image = waverec(coefficients, wavelet)

  return reconst_image


if __name__ == "__main__":
  # 读取图像
  tifPath = r".\nternalCalibration\20231031_clip.tif"
  dataset = gdal.Open(tifPath)
  projection = dataset.GetProjection()
  transform = dataset.GetGeoTransform()
  imgx, imgy = dataset.RasterXSize, dataset.RasterYSize
  tifArr = dataset.ReadAsArray(0, 0, imgx, imgy)
  tifArr[tifArr == 65536] = 0

  # 使用db4小波进行4级分解
  coefficients = wavelet_decomp(tifArr, "db4")

  # 重构图像
  reconst_image = wavelet_recon(coefficients, "db4")

