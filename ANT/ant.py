import xml.etree.ElementTree as ET
import xmlschema

class ANT:
    def __init__(self, xml):
        self.schema = xmlschema.XMLSchema("ANT/ant.xsd")
        self.xml = ET.tostring(xml, encoding='utf-8',
                               xml_declaration=True).decode('utf-8')

    def valid(self):
        try:
            self.schema.validate(self.xml)
            return True
        except:
            return False

    def indent(self, elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

def generate_ant(oracle, max_depth):
    def _generate_ant(oracle, node, depth):
        # Tag part
        tag = None
        if node is None:
            # Base case: most simple form of valid ant.xml
            tag = ET.Element('project', {
                'xmlns': 'http://example.com/yourproject',
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'
            })
        else:
            tag_name = oracle.select(1)
            tag = ET.Element(tag_name)
        # Children part
        if depth < max_depth:
            # No empty tree with just one leaf.
            num_children = oracle.select(2)
            if num_children > 0:
                for _ in range(num_children):
                    child = _generate_ant(oracle, tag, depth + 1)
                    if child is not None:
                        tag.append(child)
        return tag
    ant = ANT(_generate_ant(oracle, None, 0))
    return ant, 10, ant.valid()
