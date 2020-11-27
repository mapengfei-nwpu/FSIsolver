# Unit conversion
# See wikipedia page of International System of Units
# https://en.wikipedia.org/wiki/International_System_of_Units


class UnitConversion:
    def __int__(self):
        self.lengthUnit = "cm"
        self.timeUnit   = "s"
        self.massUnit   = "g"
    
    def pressure_other(self, p, pressure):
        conversion_table = {"Pa":1.0,"kPa":1000.0}
        if pressure in conversion_table :
            print("Pressure conversion from ", pressure)
            return conversion_table[pressure]*pressure(p)
        else :
            print("Unit of ",pressure," is not on the table.")

    def pressure(self, p, length = "m", mass = "kg", time = "s"):
        length_scale = self.lengthConversion(self.lengthUnit, length)
        mass_scale = self.massConversion(self.massUnit, mass)
        time_scale = self.timeConversion(self.timeUnit, time)
        print("Pressure converted to : ",self.lengthUnit)
        #    mass
        #--------------
        # time^2*length
        return p*mass_scale/time_scale/time_scale/length_scale
    


    def lengthConversion(self, unit_1, unit_0):
        conversion_table = {"m:cm":100.0,"cm:m":0.01,"m:mm":1000.0,"mm:m":0.001}
        a = unit_1+":"+unit_0
        if a in conversion_table:
            pass
        return 100
        
    def massConversion(self, unit_1, unit_0):
        return 100
    
    def timeConversion(self, unit_1, unit_0):
        return 100
