import unittest
import numpy as np
from mmd.integrals.onee import _boys as boys 

class test_util(unittest.TestCase):
    def test_boys(self):
        ''' Values from 
            Guseinov II, Mamedov BA. J. Mat. Chem. 2006 Aug 1;40(2):179-83.
            using their Mathematica results
        '''
        places = 12
        self.assertAlmostEqual(boys(0.5,6.8),7.34475165333247E-02,places=places)
        self.assertAlmostEqual(boys(13,14.1),1.56775160456192E-07,places=places)
        self.assertAlmostEqual(boys(20.6,32.4),2.17602798734846E-14,places=places)
        self.assertAlmostEqual(boys(25,6.4),4.28028518677348E-05,places=places)
        self.assertAlmostEqual(boys(64,50),5.67024356263279E-24,places=places)
        self.assertAlmostEqual(boys(75.5,40.8),2.63173492081630E-20,places=places)
        self.assertAlmostEqual(boys(80.3,78.2),6.35062774057122E-36,places=places)
        self.assertAlmostEqual(boys(4,7),8.03538503977806E-04,places=places)
        self.assertAlmostEqual(boys(8.5,3.6),2.31681539108704E-03,places=places)
        self.assertAlmostEqual(boys(15.3,20.7),5.40914879973724E-10,places=places)
        self.assertAlmostEqual(boys(1.8,25.3),3.45745419193244E-04,places=places)
        self.assertAlmostEqual(boys(30,26),3.57321060811178E-13,places=places)
        self.assertAlmostEqual(boys(46.8,37.6),1.91851951160577E-18,places=places)
        self.assertAlmostEqual(boys(100,125.1),7.75391047694625E-55,places=places)
        ''' Values from 
            Mamedov BA. J. Mat. Chem. 2004 Jul 1;36(3):301-6.
            Using Eq(3) values
        '''
        # Table 1
        self.assertAlmostEqual(boys(8,16 ),4.02308592502660E-07,places=places)   
        self.assertAlmostEqual(boys(15,27),1.08359515555596E-11,places=places)
        # error in text? should be E-13 not E-03
        self.assertAlmostEqual(boys(20,30),1.37585444267909E-13,places=places)
        self.assertAlmostEqual(boys(25,13),8.45734447905704E-08,places=places)
        self.assertAlmostEqual(boys(31,34),2.90561943091301E-16,places=places)
        self.assertAlmostEqual(boys(11,38),4.04561442253925E-12,places=places)
        self.assertAlmostEqual(boys(42,32),5.02183610419087E-16,places=places)
        self.assertAlmostEqual(boys(75,30),1.01429517438537E-15,places=places)
        self.assertAlmostEqual(boys(100,33),3.42689684943483E-17,places=places)
        self.assertAlmostEqual(boys(20,1.4E-3),2.43577075309547E-02,places=places)
        self.assertAlmostEqual(boys(45,6.4E-5),1.09883228385254E-02,places=places)
        self.assertAlmostEqual(boys(100,2.6E-7),4.97512309732144E-03,places=places)
        # Table 2
        self.assertAlmostEqual(boys( 8, 42), 1.11826597752251E-10,places=places) 
        self.assertAlmostEqual(boys(16, 50), 2.40509456111904E-16,places=places)
        self.assertAlmostEqual(boys(21, 56), 1.43739730342730E-19,places=places)
        self.assertAlmostEqual(boys(12, 60), 4.05791663779760E-15,places=places)
        self.assertAlmostEqual(boys(15, 53), 3.14434039868936E-16,places=places)
        self.assertAlmostEqual(boys(18, 58), 1.78336953967902E-18,places=places)
        # Table 3
        self.assertAlmostEqual(boys(  8,  63), 3.56261924865627E-12,places=places) 
        self.assertAlmostEqual(boys( 14,  68), 3.09783511327517E-17,places=places) 
        self.assertAlmostEqual(boys( 20,  73), 1.71295886102040E-21,places=places) 
        self.assertAlmostEqual(boys( 33,  85), 1.74268831008018E-29,places=places) 
        self.assertAlmostEqual(boys( 36, 100), 3.08919970425521E-33,places=places) 
        self.assertAlmostEqual(boys(100, 120), 4.97723065221079E-53,places=places) 
        # Table 4
        self.assertAlmostEqual(boys( 5.7 ,    13.3 ), 9.02296149898981E-06,places=places) 
        self.assertAlmostEqual(boys( 0.5 ,    23.6 ), 2.11864406767677E-02,places=places)
        self.assertAlmostEqual(boys( 23.8,    3.4  ), 7.92593349658604E-04,places=places) 
        self.assertAlmostEqual(boys( 25.8,    0.4  ), 1.29331240687006E-02,places=places) 
        self.assertAlmostEqual(boys( 28.3,    0.002), 1.73275865107165E-02,places=places) 
        # error in paper? seems they transposed exponent 
        self.assertAlmostEqual(boys( 36.6,   42.7  ), 7.12651246345736E-20,places=places) 
        self.assertAlmostEqual(boys( 43.2,   54.2  ), 1.53021328677383E-24,places=places) 
        self.assertAlmostEqual(boys( 64.3,  75.4   ), 5.52165865464571E-34,places=places) 
        self.assertAlmostEqual(boys(104.6, 115.4   ), 1.26350192129925E-51,places=places) 
        self.assertAlmostEqual(boys(115.6,   5.4   ), 2.03911971791491E-05,places=places) 

 


