import logging

from fpvgcc import fpv, gccMemoryMap
from fpvgcc.datastructures import ntree


class GCCMemoryMapNode(gccMemoryMap.GCCMemoryMapNode):
    _leaf_property = '_size'
    _ident_property = 'name'
    ident = None

    @property
    def name(self):
        return self.ident

    @name.setter
    def name(self, value):
        self.ident = value

    @property
    def _is_ident_property_set(self):
        return self.ident is not None

    @property
    def _is_leaf_property_set(self):
        return self._size is not None

    def add_child(self, newchild=None, name=None,
                  address=None, size=None, fillsize=0,
                  arfile=None, objfile=None, arfolder=None):
        if newchild is None:
            nchild = GCCMemoryMapNode(name=name, address=None, size=None,
                                      fillsize=0, arfile=None, objfile=None,
                                      arfolder=None)
            newchild = super().add_child(nchild)
        else:
            newchild = super().add_child(newchild)
        return newchild


def patch_fpv():
    fpv.logging = gccMemoryMap.logging = ntree.logging = logging.getLogger('fpv')
    fpv.logging.setLevel(logging.CRITICAL)
    gccMemoryMap.GCCMemoryMap.node_t = GCCMemoryMapNode
