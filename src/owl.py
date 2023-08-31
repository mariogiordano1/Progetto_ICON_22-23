from owlready2 import *
onto = get_ontology("http://aqi.org/")
#onto = get_ontology("file:///Users/patrickclark/Desktop/Projects/OWL/Onto/onto.owl").load()

print(list(onto.classes()))

with onto:
    class Observations(Thing):
        pass
    class Pollutant(Thing):
        pass
    class WeatherCondition(Thing):
        pass
    class AQI(Thing):
        pass
    class AirPollutants(Pollutant):
        pass
    class DirectContributor(AirPollutants):
        pass
    class IndirectContributor(AirPollutants):
        pass
    class GaseousPollutant(DirectContributor):
        pass
    class SulfurOxides(GaseousPollutant):
        pass
    class SulfurDioxide(SulfurOxides):
        pass
    class SulfurTrioxide(SulfurOxides):
        pass
    class NitrogenOxides(GaseousPollutant):
        pass
    class NitrogenMonoxide(NitrogenOxides):
        pass
    class NitrogenDioxide(NitrogenOxides):
        pass
    class CarbonOxides(GaseousPollutant):
        pass
    class CarbonMonoxide(CarbonOxides):
        pass
    class CarbonDioxide(CarbonOxides):
        pass
    class ParticulateMatters(DirectContributor):
        pass
    class ParticulateMatter2_5(ParticulateMatters):
        pass
    class ParticulateMatter10(ParticulateMatters):
        pass
    class Ozone(GaseousPollutant):
        pass
    
    '''
    Properties, ObjectProperties mapped to xsd:Float, even though technically incorrect, needed to map domains for a better representation
    '''    
    class PM25Concentration(ParticulateMatter2_5 >> float, FunctionalProperty):
        pass
    class PM10Concentration(ParticulateMatter10 >> float, FunctionalProperty):
        pass
    class OzoneConcentration(Ozone >> float, FunctionalProperty):
        pass
    class NO2Concentration(NitrogenDioxide >> float, FunctionalProperty):
        pass
    class SO2Concentration(SulfurDioxide >> float, FunctionalProperty):
        pass
    class SO3Concentration(SulfurTrioxide >> float, FunctionalProperty):
        pass
    class COConcentration(CarbonMonoxide >> float, FunctionalProperty):
        pass
    class CO2Concentration(CarbonDioxide >> float, FunctionalProperty):
        pass
    class AQIValue(AQI >> float, FunctionalProperty):
        pass
       
print(list(onto.classes()))
onto.save(file = "AQI.rdf", format = "rdfxml")