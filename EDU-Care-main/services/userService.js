const userModel = require("../models/userModel");
const slugify = require("slugify");
const asyncHandler = require("express-async-handler");
const factory = require("./handlersFactory");
const { v4: uuidv4 } = require("uuid");
const sharp = require("sharp");

const ApiError = require("../utils/ApiError");
const bcrypt = require("bcrypt");

// Image processing

//@desc get list of brands
//@route get/ /api/v1/brands
//@access public
exports.getUsers = factory.getMany(userModel);

//desc     get sepsfic brand  by id
//@route    get /v1/brands/:id

exports.getUser = factory.getOne(userModel);

// desc  create brand
//@route  post/ /api/v1/brands
//@access  private
exports.createUser = factory.createOne(userModel);
//@desc     update sepcfic brand by id
//@route    put / api/v1/brand/:id
//@access   private
exports.updateUser = factory.updateOne(userModel);
//to change password or update it


exports.deleteUser = factory.deleteOne(userModel);
