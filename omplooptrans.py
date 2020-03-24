#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import collections
import itertools


class Token:
    def __init__(self,s,kind=None):
        if type(self) == Token and kind==None:
            raise Exception("kind required")
        self.s = s
        self.kind=kind

    def __str__(self):
        return self.s

    def __repr__(self):
        return f"<{type(self).__name__}:{self.s!r}>"

    def __eq__(self,other):
        if type(self)!= type(other):
            return False
        return self.issame(other)

    def issame(self,other):
        if type( self)  == Token and self.kind != other.kind:
            return False
        return self.s == other.s


class IdentifierToken(Token):
    pass

class IntegerToken(Token):
    def __init__(self,s,kind=None,val=None):
        super(IntegerToken, self).__init__(s,kind)
        if val is None:
            val = int(s)
        self.val = val

    def issame(self,other):
        return self.val == other.val


class KeywordToken(Token):
    def __init__(self,keyword ,s=None,kind=None):
        if s ==None:
            s = keyword
        assert  keyword==s
        super(KeywordToken, self).__init__(s,kind)

    def issame(self,other):
        return True


class ForToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(ForToken, self).__init__('for',s,kind)

class LParenToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(LParenToken, self).__init__('(',s,kind)

class RParenToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(RParenToken, self).__init__(')',s,kind)

class LBraceToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(LBraceToken, self).__init__('{',s,kind)

class RBraceToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(RBraceToken, self).__init__('}',s,kind)

class LAngleToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(LAngleToken, self).__init__('<',s,kind)

class RAngleToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(RAngleToken, self).__init__('>',s,kind)

class AssignToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(AssignToken, self).__init__('=',s,kind)

class AssignAddToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(AssignAddToken, self).__init__('+=',s,kind)

class SemicolonToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(SemicolonToken, self).__init__(';',s,kind)

class CommaToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(CommaToken, self).__init__(',',s,kind)

class PlusPlusToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(PlusPlusToken, self).__init__('++',s,kind)

class PragmaToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(PragmaToken, self).__init__('#pragma',s,kind)

class EndOfPragmaToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        super(EndOfPragmaToken, self).__init__('\n',s,kind)

class EofToken(KeywordToken):
    def __init__(self,s=None,kind=None):
        self.s =None

def lex(input):
    tokenkinds = [
      ('for',r'for',ForToken,True,False),
      ('pragma',r'\#pragma',PragmaToken,True,False),
      ('integer', r'\d+?',IntegerToken,True,True),
      ('identifier', r'[A-Za-z_]\w*',IdentifierToken,True,True),
      ('plusplus', r'\+\+',PlusPlusToken,True,True),
      ('assignadd', r'\+\=',AssignAddToken,True,True),
      ('eq', r'\=\=',Token,True,True),
      ('lparen', r'\(',LParenToken,True,True),
      ('rparen', r'\)',RParenToken,True,True),
      ('lbrace', r'\{',LBraceToken,True,True),
      ('rbrace', r'\}',RBraceToken,True,True),
      ('semicolon', r'\;',SemicolonToken,True,True),
      ('comma', r'\,',CommaToken,True,True),
      ('plus', r'\+',Token,True,True),
      ('langle', r'\<',LAngleToken,True,True),
      ('rangle', r'\>',RAngleToken,True,True),
      ('assign', r'\=',AssignToken,True,True),
      ]
    kindlookup = { name: clazz for (name,regex,clazz,normalmode,pragmamode)  in tokenkinds}

    regex_normalmode = '|'.join(r'(?P<{name}>{regex})'.format(name=name,regex=regex) for (name,regex,clazz,normalmode,pragmamode) in tokenkinds if normalmode) + r'|\s+'
    get_token_normalmode = re.compile(regex_normalmode).match

    regex_pragmamode = '|'.join(r'(?P<{name}>{regex})'.format(name=name,regex=regex) for (name,regex,clazz,normalmode,pragmamode) in tokenkinds if pragmamode) + r'|\s+'
    get_token_pragmamode = re.compile(regex_pragmamode).match

    while True:
        s = input.readline()
        if len(s)==0:
            break

        pos = 0
        while pos < len(s):
            startpos = pos
            tok = get_token_normalmode(s,startpos)
            if tok is None:
                raise Exception("Unrecognized token")
            kind = tok.lastgroup
            pos = tok.end()
            if not kind:
                # Ignorable token (whitespace, comment, ...)
                continue
            tokstr = tok[0]
            clazz = kindlookup[kind]
            yield clazz(tokstr,kind)

            # Preprocessor parsing mode
            # No line-continuation supported
            if clazz == PragmaToken:
                while pos < len(s):
                    startpos = pos
                    tok = get_token_pragmamode(s,startpos)
                    if tok is None:
                        raise Exception("Unrecognized token")
                    kind = tok.lastgroup
                    pos = tok.end()
                    if not kind:
                        # Ignorable token (whitespace, comment, ...)
                        continue
                    tokstr = tok[0]
                    clazz = kindlookup[kind]
                    yield clazz(tokstr,kind)

                yield EndOfPragmaToken()
                break
    yield EofToken()

def tokstr(*toks):
    return ' '.join(str(t) for t in toks)


class TokenStream:
    class TokenRange:
        def __init__(self,strm,start,stop):
            self.strm  = strm
            self.start = start
            self.stop =stop

        def advance(self):
            self.start += 1
            assert self.stop == None or self.start < self.stop
            if self.strm.root == self:
                self.strm.gc()

        def get_token(self):
            return self.strm.get_token(self.start)

        def is_token(self,*expected):
            tok = self.get_token()
            for e in expected:
                if isinstance(e,Token):
                    if tok == e:
                        return tok
                else:
                    for elt in e:
                        if tok == elt:
                            return tok
            return None

        def is_kind(self,*clazz):
            tok = self.get_token()
            for e in clazz:
                if issubclass(e,Token):
                    if isinstance(tok , e):
                        return tok
                else:
                    for elt in e:
                        if isinstance(tok , e):
                            return tok
            return None

        def is_identifier(self,s):
            return self.is_token(IdentifierToken(s))

        def try_token(self,*expected):
            if tok := self.is_token(*expected):
                self.advance()
                return tok
            return None

        def try_kind(self,*clazz):
            if tok := self.is_kind(*clazz):
                self.advance()
                return tok
            return None

        def expect_token(self,expected):
            tok = self.get_token()
            if tok != expected:
                raise Exception("unexpected token")
            self.advance()
            return tok

        def expect_kind(self,clazz):
            tok = self.get_token()
            if not isinstance(tok,clazz):
                raise Exception("unexpected token")
            self.advance()
            return tok

        def expect_identifier(self,s):
            return self.expect_token(IdentifierToken(s))

        def skip_until(self,expected,inclusive=False):
            while True:
                tok = self.get_token()
                if tok == expected:
                    break
                self.advance()
            if inclusive:
                self.advance()

        def as_list(self):
            return  self.strm.as_list(start=self.start,stop=self.stop)


        def slice(self,start=None,stop=None):
            if start==None:
                start=self.start
            if stop==None:
                stop=self.stop
            assert self.start <= start
            assert not self.stop or stop <= self.stop
            assert not stop or start < stop
            return TokenStream.TokenRange(strm=self.strm,start=start,stop=stop)

        def commit(self, substrm):
            assert self.start <= substrm.start
            assert not self.stop or substrm.stop <= self.stop
            assert not substrm.stop or substrm.start<= substrm.stop
            self.start  = substrm.start
            self.stop = substrm.stop

            if self.strm.root == self:
                self.strm.gc()


    def __init__(self,tokstream):
        self.tokstream = tokstream
        self.stack_offset = 0
        self.stack = collections.deque()
        self.root = TokenStream.TokenRange(strm=self,start=0,stop=None)

    def load(self,max_pos):
        while self.stack_offset + len(self.stack) <= max_pos:
            tok = next(self.tokstream)
            self.stack.append(tok)

    def get_token(self,pos):
        self.load(pos)
        return self.stack[pos - self.stack_offset]

    def gc(self):
        while self.stack_offset < self.root.start:
            self.stack.popleft()
            self.stack_offset += 1

    def range(self):
        return self.root

    def as_list(self,start,stop):
        self.load(max_pos=stop-1)
        return list( itertools.islice(self.stack,start-self.stack_offset,stop-self.stack_offset))


















class ASTNode:
    pass

class Stmt(ASTNode):
    pass


class Type(ASTNode):
    def __init__(self,name):
        self.name=name

    def __str__(self):
        return self.name

class DeclStmt(Stmt):
    def __init__(self,ty,declared,initializer):
        self.ty = ty
        self.declared = declared
        self.initializer = initializer

    def __str__(self):
        if self.initializer:
            return f"{self.ty} {self.declared} = {self.initializer};"
        return f"{self.ty} {self.declared};"

class BlockStmt(Stmt):
    def __init__(self,stmts):
        self.stmts = stmts

    def __str__(self):
        lines = ['{']
        for stmt in self.stmts:
            lines .append( '  '+ str(stmt).replace('\n','\n  '))
        lines.append('}')
        return '\n'.join(lines)

    def transform(self):
        nstmts = []
        for stmt in self.stmts:
            nstmts.append(stmt.transform())
        return BlockStmt(nstmts)



class ForStmt(Stmt):
    def __init__(self,init,cond,incr,body):
        self.init = init
        self.cond=cond
        self.incr=incr
        self.body = body

    def __str__(self):
        init = self.init
        if init == None:
            init = ';'
        cond = self.cond
        if cond==None:
            cond = ''
        incr = self.incr
        if incr==None:
            incr=''
        body = str(self.body).replace('\n', '\n  ')
        return f"for ({init} {cond}; {incr})\n  {body}"

    def analyze(self):
        assert isinstance(self.init,DeclStmt)
        indvarty = self.init.ty
        indvarname = self.init.declared
        lb = self.init.initializer

        assert isinstance(self.cond,LtExpr)
        assert isinstance(self.cond.lhs,VarRefExpr)
        assert self.cond.lhs.varname == indvarname
        ub = self.cond.rhs

        if isinstance(self.incr, PreIncrExpr):
            assert isinstance(self.incr, PreIncrExpr)
            assert isinstance(self.incr.arg,VarRefExpr)
            assert self.incr.arg.varname  == indvarname
            step = 1
        elif isinstance(self.incr, AssignAddExpr):
            assert isinstance(self.incr.lhs,VarRefExpr)
            assert self.incr.lhs.varname == indvarname
            assert isinstance(self.incr.rhs,IntegerLiteralExpr)
            step = self.incr.rhs.val
        else:
            raise Exception("Incompatible incr expression")

        return (lb,ub,step,indvarty,indvarname)

    def transform(self):
        newbody = self.body.transform()
        return ForStmt(init=self.init, cond=self.cond,incr=self.incr,body=newbody)


class OtherStmt(Stmt):
    def __init__(self,toks):
        self.toks = toks

    def __str__(self):
        return tokstr(*self.toks)

    def transform(self):
        return self


class Expr(ASTNode):
    pass

class IntegerLiteralExpr(Expr):
    def __init__(self,val):
        self.val = val

    def __str__(self):
        return str(self.val)

class VarRefExpr(Expr):
    def __init__(self,varname):
        self.varname = varname

    def __str__(self):
        return self.varname

class UnaryOpExpr(Expr):
    def __init__(self,arg,op=None):
        self.arg=arg

class PreIncrExpr(UnaryOpExpr):
    pass

class BinaryOpExpr(Expr):
    def __init__(self,lhs,rhs):
        self.lhs=lhs
        self.rhs=rhs

    def __str__(self):
        return f"({self.lhs} {self.infix} {self.rhs})"

class LtExpr(BinaryOpExpr):
    infix = '<'

class AddExpr(BinaryOpExpr):
    infix = '+'

class AssignAddExpr(BinaryOpExpr):
    infix = '+='

class CallExpr(Expr):
    def __init__(self,funcname,*args):
        self.funcname=funcname
        self.args=args

    def __str__(self):
        args = ', '.join(str(a) for a in self.args)
        return f"{self.funcname}({args})"


class OMPClause(ASTNode):
    def __init__(self,name):
        self.name = name

class OMPSizesClause(OMPClause):
    def __init__(self,sizes):
        super(OMPSizesClause,self).__init__( 'sizes')
        self.sizes = sizes

    def __str__(self):
        return f"sizes({','.join(str(e) for e in self.sizes )})"

class OtherOMPClause(OMPClause):
    def __init__(self,name,toks):
        super(OtherOMPClause,self).__init__(name)
        self.toks = toks

    def __str__(self):
        if self.toks == None:
            return self.name
        return f"{self.name}({tokstr(*self.toks)})"


class OMPLoopDirective(Stmt):
    def __init__(self,name,clauses,nest):
        self.name=name
        self.clauses =clauses
        self.nest = nest

    def __str__(self):
        clauses = ''.join(' ' +str(c) for c in self.clauses)
        return f"#pragma omp {self.name}{clauses}\n{self.nest}"

    def clauses_of_kind(self,clazz):
        return (c for c in self.clauses if isinstance(c,clazz) )

class OtherOMPLoopDirective(OMPLoopDirective):
    def __init__(self,name,clauses,nest):
        super(OtherOMPLoopDirective,self).__init__(name=name,clauses=clauses,nest=nest)

    def transform(self):
        newnest = self.nest.transform()
        return OtherOMPLoopDirective(name=self.name,clauses=self.clauses,nest=newnest)


class OMPLoopTransformationDirective(OMPLoopDirective):
    def __init__(self,name,clauses,nest):
        super(OMPLoopTransformationDirective,self).__init__(name=name,clauses=clauses,nest=nest)


class OMPTileDirective(OMPLoopTransformationDirective):
    def __init__(self,clauses,nest):
        super(OMPTileDirective,self).__init__(name='tile',clauses=clauses,nest=nest)

    def transform(self):
        (sizes,) = self.clauses_of_kind(OMPSizesClause)
        sizes = sizes.sizes
        count = len(sizes)

        loops = []
        cur = self.nest.transform()
        for depth in range(count):
            loops .append(cur)
            cur = cur.body
        body = cur


        floorloops = [None]*count
        tileloops =[None]*count

        for d,loop in enumerate(loops):
            assert isinstance(sizes[d],IntegerLiteralExpr)
            tilesize = sizes[d].val

            (lb,ub,step,ty,indvar) = loop.analyze()

            floorname = '__' + indvar + '_floor_iv'
            tilename = indvar

            floorinit = DeclStmt(ty,floorname,lb)
            floorref = VarRefExpr(floorname)
            floorcond = LtExpr(floorref, ub)
            floorincr = AssignAddExpr(floorref,IntegerLiteralExpr (step*tilesize))

            tileinit = DeclStmt(ty,tilename,floorref)
            tileref = VarRefExpr(floorname)
            tilecond = LtExpr(floorref, CallExpr('min', AddExpr(floorref,tilesize), ub))
            tileincr = AssignAddExpr(floorref, IntegerLiteralExpr(step))

            floorloops [d] = ForStmt(floorinit,floorcond,floorincr,None)
            tileloops[d] = ForStmt(tileinit,tilecond,tileincr,None)

        for d in range(count-1):
            floorloops[d].body = floorloops[d+1]
            tileloops[d].body = tileloops[d+1]
        floorloops[-1].body = tileloops[0]
        tileloops[-1].body = body

        return floorloops[0]




def try_parse_expr_rhs(lhs, strm):
    if strm.try_kind(LAngleToken):
        rhs = parse_expr(strm)
        return LtExpr(lhs,rhs)
    elif strm.try_kind(AssignAddToken):
        rhs = parse_expr(strm)
        return AssignAddExpr(lhs,rhs)

    return None


def parse_unary_expr(strm):
    if valtok := strm.try_kind(IntegerToken):
        return IntegerLiteralExpr(valtok.val)
    elif idtok := strm.try_kind(IdentifierToken):
        return VarRefExpr(idtok.s)
    elif strm.try_kind(PlusPlusToken):
        arg = parse_unary_expr(strm)
        return PreIncrExpr(arg)

    raise Exception("Syntax error / unimplemented expression")

def parse_expr(strm):
    lhs = parse_unary_expr(strm)

    if expr := try_parse_expr_rhs(lhs,strm):
        return expr

    return lhs





try_parse_type_known_types = [IdentifierToken('int'),IdentifierToken('long'),IdentifierToken('int64_t'),IdentifierToken('unsigned'),IdentifierToken('short')]
def try_parse_type(strm):
    if tok := strm.try_token(try_parse_type_known_types):
        return Type(name=tok.s)
    return None

def parse_clause(strm):
    assert strm.is_kind(IdentifierToken)
    clausename = strm.expect_kind(IdentifierToken).s

    if clausename == 'sizes':
        strm.expect_kind( LParenToken)
        sizes = []
        while True:
            sizes.append( parse_expr(strm))
            if not strm.is_kind(CommaToken):
                break
            strm.advance()
        strm.expect_kind(RParenToken)
        return OMPSizesClause(sizes)
    else:
        if strm.try_kind(LParenToken):
            substrm =  strm.slice()
            substrm.skip_until(RParenToken())
            toks = strm.slice(stop=substrm.start) . as_list()
            strm.commit(substrm)
            strm.expect_kind(RParenToken)
        else:
            toks=None

        return OtherOMPClause(clausename,toks)


def parse_clauses(strm):
    clauses=[]
    while True:
        if strm.is_kind(EndOfPragmaToken):
            break
        clauses.append(parse_clause(strm))

    return clauses


def parse_executable_pragma(strm):
    strm.expect_kind(PragmaToken)
    strm.expect_identifier('omp')
    directivename = strm.expect_kind(IdentifierToken).s
    clauses = parse_clauses(strm)
    strm.expect_kind(EndOfPragmaToken)
    nest = parse_stmt(strm)

    if directivename == 'tile':
        return OMPTileDirective(clauses=clauses,nest=nest)
    else:
        return OtherOMPLoopDirective(name=directivename,clauses=clauses,nest=nest)


def parse_block_stmt(strm):
    stmts = []
    strm.expect_kind(LBraceToken)
    while True:
        if strm.is_kind(RBraceToken):
            break
        stmts.append( parse_stmt(strm))

    strm.expect_kind(RBraceToken)
    return BlockStmt(stmts)

def parse_for_stmt(strm):
    strm.expect_kind(ForToken)
    strm.expect_kind(LParenToken)
    initstmt = parse_stmt(strm)
    # separator semicolon parsed as part of initstmt
    condexpr = parse_expr(strm)
    strm.expect_kind(SemicolonToken)
    increxpr = parse_expr(strm)
    strm.expect_kind(RParenToken)
    body = parse_stmt(strm)
    return ForStmt(init=initstmt,cond=condexpr,incr=increxpr,body=body)


def parse_decl_stmt(ty,strm):
    declname = strm.expect_kind(IdentifierToken).s
    initexpr = None
    if tok := strm.try_kind(AssignToken):
        initexpr = parse_expr(strm)
    strm.expect_kind(SemicolonToken)
    return DeclStmt(ty=ty,declared=declname,initializer=initexpr)

def parse_other_stmt(strm):
    substrm = strm.slice()
    substrm.skip_until(SemicolonToken(),inclusive=True)
    toks = strm.slice(stop=substrm.start).as_list()
    strm.commit(substrm)
    return OtherStmt(toks)


def parse_stmt(strm):
    if strm.is_kind(PragmaToken):
        return parse_executable_pragma(strm)
    elif strm.is_kind(LBraceToken):
        return parse_block_stmt(strm)
    elif strm.is_kind(ForToken):
        return parse_for_stmt(strm)
    elif ty := try_parse_type(strm):
        return parse_decl_stmt(ty,strm)
    else:
        return parse_other_stmt(strm)



def main():
    parser = argparse.ArgumentParser(description="OpenMP loop transformation simulator")
    parser.add_argument('input', type=argparse.FileType('r'),default='-')
    parser.add_argument('-o', dest='output', default='-', type=argparse.FileType('w+'))
    parser.add_argument('--debug-lex',action='store_true', help="Print token stream")

    args = parser.parse_args()

    tokstream = lex(args.input)
    def print_and_yield(stream):
      for x in stream:
        print(repr(x))
        yield x
    if args.debug_lex:
        tokstream = print_and_yield(tokstream)
    strm = TokenStream(tokstream).range()

    stmts = []
    while True:
        if strm.is_kind(EofToken):
            break
        stmt = parse_stmt(strm)
        transformed = stmt.transform()
        print(transformed,file=args.output)

    return 0


if __name__ == '__main__':
    if errcode := main():
        exit(errcode)
