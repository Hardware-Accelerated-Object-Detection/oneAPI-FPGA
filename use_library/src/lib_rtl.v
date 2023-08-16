`timescale 1 ps / 1 ps

module lib_rtl (
  input   clock,
  input   resetn,
  input   ivalid,
  input   iready,
  output  ovalid,
  output  oready,
  input   [31:0]  datain1,
  input   [31:0]  datain2,
  output  [31:0]  dataout);

  logic [31:0] dataout_buffer, dataout_reg;
  assign ovalid = ivalid;
  assign oready = iready;
  assign dataout = dataout_reg;

  always@(posedge clock or negedge resetn) begin
    // reset
    if(!resetn) begin
      dataout_reg <= 0;
      dataout_buffer <= 0;  
    end
  end

  // output to downstream
  always@(posedge clock) begin
    if(!ivalid) begin
      dataout_buffer <= 0;
      dataout_reg <= 0;
    end
    else begin
      dataout_buffer <= dataout_reg;
      dataout_reg <= datain1 + datain2;
    end  
  end

endmodule

