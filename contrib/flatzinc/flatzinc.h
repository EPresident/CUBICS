/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *     Guido Tack <tack@gecode.org>
 *
 *  Copyright:
 *     Guido Tack, 2007
 *
 *  Last modified:
 *     $Date: 2010-07-02 19:18:43 +1000 (Fri, 02 Jul 2010) $ by $Author: tack $
 *     $Revision: 11149 $
 *
 *  This file is part of Gecode, the generic constraint
 *  development environment:
 *     http://www.gecode.org
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef __GECODE_FLATZINC_HH__
#define __GECODE_FLATZINC_HH__

#include <iostream>
#include <map>
#include <cassert>

#include <flatzinc/conexpr.h>
#include <flatzinc/ast.h>
#include <flatzinc/varspec.h>

#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>

/**
 * \namespace FlatZinc
 * \brief Interpreter for the %FlatZinc language
 *
 * The Gecode::FlatZinc namespace contains all functionality required
 * to parse and solve constraint models written in the %FlatZinc language.
 *
 */

namespace FlatZinc {

  class BoolVar {};
  class SetVar {};

  class Printer;

  /**
   * \brief A space that can be initialized with a %FlatZinc model
   *
   */
  class FlatZincModel {
  public:
    enum Meth {
      SAT, //< Solve as satisfaction problem
      MIN, //< Solve as minimization problem
      MAX  //< Solve as maximization problem
    };
  protected:
    /// Number of integer variables
    int intVarCount;
    /// Number of Boolean variables
    int boolVarCount;
    /// Number of set variables
    int setVarCount;

    /// Index of the integer variable to optimize
    int _optVar;
  
    /// Whether to solve as satisfaction or optimization problem
    Meth _method;
    
    /// Annotations on the solve item
    AST::Array* _solveAnnotations;

  public:
    /// The integer variables
    IntVariables* intVariables;
    /// Indicates whether an integer variable is introduced by mzn2fzn
    std::vector<bool> iv_introduced;
    /// Indicates whether an integer variable aliases a Boolean variable
    std::vector<int> iv_boolalias;
    /// The Boolean variables
    std::vector<BoolVar> bv;
    /// Indicates whether a Boolean variable is introduced by mzn2fzn
    std::vector<bool> bv_introduced;
    /// The set variables
    std::vector<SetVar> sv;
    /// Indicates whether a set variable is introduced by mzn2fzn
    std::vector<bool> sv_introduced;
    /// The integer constraints
    IntConstraints* intConstraints;

    /// Construct empty space
    FlatZincModel(void);
  
    /// Destructor
    ~FlatZincModel(void);
  
    /// Initialize space with given number of variables
    void init(int intVars, int boolVars, int setVars);

    /// Create new integer variable from specification
    void newIntVar(IntVarSpec* vs);
    /// Link integer variable \a iv to Boolean variable \a bv
    void aliasBool2Int(int iv, int bv);
    /// Return linked Boolean variable for integer variable \a iv
    int aliasBool2Int(int iv);
    /// Create new Boolean variable from specification
    void newBoolVar(BoolVarSpec* vs);
    /// Create new set variable from specification
    void newSetVar(SetVarSpec* vs);
  
    /// Post a constraint specified by \a ce
    void postConstraint(const ConExpr& ce, AST::Node* annotation);
  
    /// Post the solve item
    void solve(AST::Array* annotation);
    /// Post that integer variable \a var should be minimized
    void minimize(int var, AST::Array* annotation);
    /// Post that integer variable \a var should be maximized
    void maximize(int var, AST::Array* annotation);

    /// Run the search
    void run(std::ostream& out, const Printer& p);
  
    /// Produce output on \a out using \a p
    void print(std::ostream& out, const Printer& p) const;

    /// Return whether to solve a satisfaction or optimization problem
    Meth method(void) const;

    /// Return index of variable used for optimization
    int optVar(void) const;

    /**
     * \brief Create branchers corresponding to the solve item annotations
     *
     * If \a ignoreUnknown is true, unknown solve item annotations will be
     * ignored, otherwise a warning is written to \a err.
     */
    void createBranchers(AST::Node* ann, bool ignoreUnknown,
                         std::ostream& err = std::cerr);

    /// Return the solve item annotations
    AST::Array* solveAnnotations(void) const;

  };

  /**
   * \brief Output support class for %FlatZinc interpreter
   *
   */
  class Printer {
  private:
    AST::Array* _output;
    void printElem(std::ostream& out,
                   AST::Node* ai, const FlatZincModel& m) const;
  public:
    Printer(void) : _output(NULL) {}
    void init(AST::Array* output);

    void print(std::ostream& out, const FlatZincModel& m) const;
  
    ~Printer(void);
    
  private:
    Printer(const Printer&);
    Printer& operator=(const Printer&);
  };

  /// %Exception class for %FlatZinc errors
  class Error {
  private:
    const std::string msg;
  public:
    Error(const std::string& where, const std::string& what)
    : msg(where+": "+what) {}
    const std::string& toString(void) const { return msg; }
  };

  /**
   * \brief Parse FlatZinc file \a fileName into \a fzs and return it.
   *
   * Creates a new empty FlatZincModel if \a fzs is NULL.
   */
  FlatZincModel* parse(const std::string& fileName,
                       Printer& p, std::ostream& err = std::cerr,
                       FlatZincModel* fzs=NULL);

  /**
   * \brief Parse FlatZinc from \a is into \a fzs and return it.
   *
   * Creates a new empty FlatZincModel if \a fzs is NULL.
   */
  FlatZincModel* parse(std::istream& is,
                       Printer& p, std::ostream& err = std::cerr,
                       FlatZincModel* fzs=NULL);

}

#endif

// STATISTICS: flatzinc-any
