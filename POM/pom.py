import xml.etree.ElementTree as ET
import xmlschema

MAX_DEPTH = 5


class POM:
    def __init__(self, xml):
        self.schema = xmlschema.XMLSchema("POM/maven-4.0.0.xsd")
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


def generate_pom(oracle, max_depth):
    def _generate_pom(oracle, node, depth):
        # Tag part
        tag = None
        if node is None:
            # Base case: most simple form of valid pom.xml
            tag = ET.Element('project', {
                'xmlns': 'http://maven.apache.org/POM/4.0.0'
            })
            base_children = [("modelVersion", "4.0.0"), ("groupId", "com.example"),
                             ("artifactId", "my-app"), ("version", "1.0.0")]
            for t, s in base_children:
                child = ET.SubElement(tag, t)
                child.text = s
        else:
            tag_name = oracle.select(1)
            tag = ET.Element(tag_name)
        # Children part
        if depth < max_depth:
            # No empty tree with just one leaf.
            num_children = oracle.select(2)
            if num_children > 0:
                for _ in range(num_children):
                    child = _generate_pom(oracle, tag, depth + 1)
                    if child is not None:
                        tag.append(child)
            else:
                # Add text (can be also learned by oracle.select(text_options, 3))
                text = "Leaf text"
                tag.text = text
        else:
            # Add text (can be also learned by oracle.select(text_options, 3))
            text = "Leaf text"
            tag.text = text
        return tag
    pom = POM(_generate_pom(oracle, None, 0))
    return pom, 10, pom.valid()
