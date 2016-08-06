#include <boost/python.hpp>
#include <fixed/ghmcLt.hpp>
#include <fixed/ghmcLt1Acc.hpp>
#include <fixed/hmcLt.hpp>
#include <fixed/hmcLt1Acc.hpp>
#include <fixed/ighmc.hpp>
#include <fixed/ighmc1Acc.hpp>
#include <fixed/ihmc.hpp>
#include <fixed/ihmc1Acc.hpp>

BOOST_PYTHON_MODULE(fixed)
{
    using namespace boost::python;
	def ("ghmcLt", ghmcLt);
	def ("ghmcLt1Acc", ghmcLt1Acc);
	def ("hmcLt", hmcLt);
	def ("hmcLt1Acc", hmcLt1Acc);
	def ("ighmc", ighmc);
	def ("ighmc1Acc", ighmc1Acc);
	def ("ihmc", ihmc);
	def ("ihmc1Acc", ihmc1Acc);
}