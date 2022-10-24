import Cython.Build
import distutils.core
 
def py2c(file):
    cpy = Cython.Build.cythonize(file) # 返回distutils.extension.Extension对象列表
 
    distutils.core.setup(
	    name = 'pyd的编译', # 包名称
	    version = "1.0",    # 包版本号
	    ext_modules= cpy,     # 扩展模块
	    author = "kdongyi",#作者
	    author_email='kdongyi@163.com'#作者邮箱
	)
 
if __name__ == '__main__':
	
    file = "range_linear.py"
    py2c(file)
