import $file.get_func_graph

@main def exec(manifest: String) = {
  val paths = ujson.read(scala.io.Source.fromFile(manifest).mkString)("files").arr.map(_.str)
  paths.foreach { filename =>
    val begin = System.nanoTime()
    println("DEEPDFA_BEGIN " + ujson.Obj("file" -> filename).toString)
    System.out.flush()
    try {
      get_func_graph.exec(filename=filename, runOssDataflow=true,
        exportJson=true, exportCpg=true, exportDataflow=false,
        deleteAfter=true, overwrite=false)
      println("DEEPDFA_DONE " + ujson.Obj("file" -> filename,
        "status" -> "ok", "seconds" -> ((System.nanoTime()-begin)/1e9)).toString)
    } catch {
      case error: Throwable =>
        println("DEEPDFA_DONE " + ujson.Obj("file" -> filename,
          "status" -> "error", "message" -> error.toString,
          "seconds" -> ((System.nanoTime()-begin)/1e9)).toString)
    }
    System.out.flush()
  }
}
