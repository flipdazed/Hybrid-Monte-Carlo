#include <boost/python.hpp>
#include <exponential/ghmc1Acc.hpp>
#include <exponential/ghmc1AccNormalised.hpp>
// #include <exponential/ghmcLt.hpp>		// numerator=ghmcLt(b,ph,th,r);
// #include <exponential/ghmcLt1Acc.hpp>	// numerator=ghmcLt(b,ph,th,r);
// #include <exponential/ghmcLtDerived.hpp>	// numerator=ghmcLt(b,ph,th,r);
#include <exponential/ghmcNormalised.hpp>
#include <exponential/hmc.hpp>
#include <exponential/hmc1Acc.hpp>
#include <exponential/hmc1AccNormalised.hpp>
#include <exponential/hmcLt.hpp>
#include <exponential/hmcLt1Acc.hpp>
#include <exponential/hmcLtDerived.hpp>
#include <exponential/hmcNormalised.hpp>
#include <exponential/ighmc.hpp>
#include <exponential/ighmc1Acc.hpp>
#include <exponential/ihmc.hpp>
#include <exponential/ihmc1Acc.hpp>

BOOST_PYTHON_MODULE(exponential)
{
    using namespace boost::python;
	def ("ghmc1Acc", ghmc1Acc);
	def ("ghmc1AccNormalised", ghmc1AccNormalised);
	// def ("ghmcLt", ghmcLt);						// numerator=ghmcLt(b,ph,th,r);
	// def ("ghmcLt1Acc", ghmcLt1Acc);			// numerator=ghmcLt(b,ph,th,r);
	// def ("ghmcLtDerived", ghmcLtDerived);		// numerator=ghmcLt(b,ph,th,r);
	def ("ghmcNormalised", ghmcNormalised);
	def ("hmc", hmc);
	def ("hmc1Acc", hmc1Acc);
	def ("hmc1AccNormalised", hmc1AccNormalised);
	def ("hmcLt", hmcLt);
	def ("hmcLt1Acc", hmcLt1Acc);
	def ("hmcLtDerived", hmcLtDerived);
	def ("hmcNormalised", hmcNormalised);
	def ("ighmc", ighmc);
	def ("ighmc1Acc", ighmc1Acc);
	def ("ihmc", ihmc);
	def ("ihmc1Acc", ihmc1Acc);
}